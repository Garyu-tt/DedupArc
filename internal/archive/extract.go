package archive

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (r *Reader) Extract(target string) error {
	if target == "" {
		target = "."
	}
	base, err := filepath.Abs(target)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(base, 0o755); err != nil {
		return err
	}

	for _, entry := range r.Manifest.Entries {
		path, err := safeJoin(base, entry.Path)
		if err != nil {
			return err
		}
		switch entry.Type {
		case "dir":
			if err := os.MkdirAll(path, os.FileMode(entry.Mode)); err != nil {
				return err
			}
		case "file":
			if err := r.extractFile(path, entry); err != nil {
				return err
			}
		default:
			return fmt.Errorf("неизвестный тип записи %q", entry.Type)
		}
	}

	// Время каталогов ставим в конце, иначе создание файлов внутри изменит mtime.
	for i := len(r.Manifest.Entries) - 1; i >= 0; i-- {
		entry := r.Manifest.Entries[i]
		if entry.Type != "dir" {
			continue
		}
		path, err := safeJoin(base, entry.Path)
		if err != nil {
			return err
		}
		mtime := time.Unix(entry.ModifiedUnix, 0)
		_ = os.Chtimes(path, mtime, mtime)
	}
	return nil
}

func (r *Reader) extractFile(path string, entry Entry) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(entry.Mode))
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()

	var h hash.Hash
	if entry.SHA256 != "" {
		h = sha256.New()
	}
	var written int64
	if err := r.entryData(entry, func(data []byte) error {
		if _, err := f.Write(data); err != nil {
			return err
		}
		if h != nil {
			_, _ = h.Write(data)
		}
		written += int64(len(data))
		return nil
	}); err != nil {
		return fmt.Errorf("%s: %w", entry.Path, err)
	}
	if written != entry.Size {
		return fmt.Errorf("%s: размер файла не совпадает", entry.Path)
	}
	if h != nil && hex.EncodeToString(h.Sum(nil)) != entry.SHA256 {
		return fmt.Errorf("%s: SHA-256 файла не совпадает", entry.Path)
	}
	if err := f.Close(); err != nil {
		return err
	}
	mtime := time.Unix(entry.ModifiedUnix, 0)
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		return err
	}
	ok = true
	return nil
}

func safeJoin(base, stored string) (string, error) {
	stored = filepath.FromSlash(stored)
	if stored == "" || filepath.IsAbs(stored) {
		return "", fmt.Errorf("небезопасный путь в архиве: %q", stored)
	}
	clean := filepath.Clean(stored)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("небезопасный путь в архиве: %q", stored)
	}
	joined := filepath.Join(base, clean)
	rel, err := filepath.Rel(base, joined)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("путь выходит за каталог распаковки: %q", stored)
	}
	return joined, nil
}

// ExtractEntry извлекает один файл по его пути внутри архива.
// Метод нужен оболочке архиватора для обычного открытия файла двойным щелчком.
func (r *Reader) ExtractEntry(entryPath, target string) error {
	var selected *Entry
	for i := range r.Manifest.Entries {
		entry := &r.Manifest.Entries[i]
		if entry.Path == entryPath && entry.Type == "file" {
			selected = entry
			break
		}
	}
	if selected == nil {
		return fmt.Errorf("файл %q не найден в архиве", entryPath)
	}
	return r.extractFile(target, *selected)
}
