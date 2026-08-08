package session

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Garyu-tt/deduparc/internal/archive"
	"github.com/Garyu-tt/deduparc/internal/platform"
)

type fingerprintEntry struct {
	Path  string `json:"path"`
	Size  int64  `json:"size"`
	Mode  uint32 `json:"mode"`
	MTime int64  `json:"mtime"`
}

func OpenArchive(path string) error {
	archivePath, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	if _, err := archive.ReadCatalog(archivePath); err != nil {
		return fmt.Errorf("не удалось открыть архив: %w", err)
	}

	fm := platform.Find7Zip()
	if fm == "" {
		return errors.New("7-Zip File Manager не найден. Переустановите DedupArc")
	}

	stateRoot, err := platform.StateRoot()
	if err != nil {
		return err
	}
	sum := sha256.Sum256([]byte(strings.ToLower(archivePath)))
	id := hex.EncodeToString(sum[:12])
	root := filepath.Join(stateRoot, "Sessions", id)
	dataDir := filepath.Join(root, "data")
	lockPath := filepath.Join(root, "session.lock")

	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	if err := acquireLock(lockPath); err != nil {
		return err
	}
	defer func() { _ = os.Remove(lockPath) }()

	_ = os.RemoveAll(dataDir)
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}
	if err := MaterializeArchive(archivePath, dataDir); err != nil {
		return err
	}

	before, err := fingerprint(dataDir)
	if err != nil {
		return err
	}

	cmd := exec.Command(fm, dataDir)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("7-Zip File Manager завершился с ошибкой: %w", err)
	}

	// Программа, запущенная из архива, может жить дольше окна File Manager.
	// Сессию нельзя собирать обратно или очищать, пока такие процессы работают.
	platform.WaitForProcessesInPath(dataDir)

	after, err := fingerprint(dataDir)
	if err != nil {
		return err
	}
	if before == after {
		_ = os.RemoveAll(root)
		return nil
	}

	if err := saveBack(archivePath, dataDir); err != nil {
		return err
	}
	_ = os.RemoveAll(root)
	return nil
}

// MaterializeArchive разворачивает архив как обычное дерево файлов. Рабочая
// сессия использует именно эту операцию, поэтому запущенная из архива программа
// видит соседние DLL, ресурсы и каталоги по привычным относительным путям.
func MaterializeArchive(path, target string) error {
	r, err := archive.Open(path)
	if err != nil {
		return err
	}
	defer r.Close()
	return r.Extract(target)
}

func ExtractArchive(path, target string) error {
	if target == "" {
		base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		target = filepath.Join(filepath.Dir(path), base)
	}
	r, err := archive.Open(path)
	if err != nil {
		return err
	}
	defer r.Close()
	return r.Extract(target)
}

func CreateArchive(output string, inputs []string) error {
	if len(inputs) == 0 {
		return errors.New("не выбраны файлы или каталоги")
	}
	if output == "" {
		if len(inputs) == 1 {
			clean := filepath.Clean(inputs[0])
			output = filepath.Join(filepath.Dir(clean), filepath.Base(clean)+".darc")
		} else {
			output = filepath.Join(filepath.Dir(filepath.Clean(inputs[0])), "Archive.darc")
		}
	}
	if !strings.EqualFold(filepath.Ext(output), ".darc") {
		output += ".darc"
	}

	return platform.WithProgress("DedupArc — создание архива", func(update platform.ProgressUpdateFunc) error {
		if !update(platform.ProgressUpdate{Stage: "Подготовка файлов", Detail: "Сканирование выбранных данных"}) {
			return archive.ErrCancelled
		}
		progress := newCreateProgressMapper()
		_, err := archive.Create(output, inputs, archive.CreateOptions{Progress: func(p archive.CreateProgress) bool {
			return update(progress(p))
		}})
		if err != nil {
			return err
		}
		if !update(platform.ProgressUpdate{Stage: "Проверка", Detail: "Проверка структуры архива", Current: 0, Total: 1}) {
			return archive.ErrCancelled
		}
		if _, err := archive.ReadCatalog(output); err != nil {
			return fmt.Errorf("созданный архив не прошёл проверку структуры: %w", err)
		}
		update(platform.ProgressUpdate{Stage: "Готово", Detail: filepath.Base(output), Current: 1, Total: 1})
		return nil
	})
}

func saveBack(original, dataDir string) error {
	return platform.WithProgress("DedupArc — сохранение изменений", func(update platform.ProgressUpdateFunc) error {
		dir := filepath.Dir(original)
		temp := filepath.Join(dir, "."+filepath.Base(original)+".new-"+strconv.FormatInt(time.Now().UnixNano(), 10))
		inputs, err := topLevelInputs(dataDir)
		if err != nil {
			return err
		}
		progress := newCreateProgressMapper()
		if _, err := archive.Create(temp, inputs, archive.CreateOptions{Progress: func(p archive.CreateProgress) bool {
			return update(progress(p))
		}}); err != nil {
			return err
		}

		if !update(platform.ProgressUpdate{Stage: "Проверка", Detail: "Проверка сохранённой версии", Current: 0, Total: 1}) {
			_ = os.Remove(temp)
			return archive.ErrCancelled
		}
		r, err := archive.Open(temp)
		if err != nil {
			_ = os.Remove(temp)
			return err
		}
		_, verifyErr := r.Verify()
		_ = r.Close()
		if verifyErr != nil {
			_ = os.Remove(temp)
			return fmt.Errorf("новая версия архива не прошла проверку: %w", verifyErr)
		}

		backup := original + ".bak"
		_ = os.Remove(backup)
		if err := os.Rename(original, backup); err != nil {
			_ = os.Remove(temp)
			return err
		}
		if err := os.Rename(temp, original); err != nil {
			_ = os.Rename(backup, original)
			_ = os.Remove(temp)
			return err
		}
		_ = os.Remove(backup)
		update(platform.ProgressUpdate{Stage: "Готово", Detail: filepath.Base(original), Current: 1, Total: 1})
		return nil
	})
}

func newCreateProgressMapper() func(archive.CreateProgress) platform.ProgressUpdate {
	started := time.Now()
	return func(p archive.CreateProgress) platform.ProgressUpdate {
		stage := "Сжатие"
		switch p.Phase {
		case archive.PhasePreparing:
			stage = "Подготовка файлов"
		case archive.PhaseWriting:
			stage = "Запись архива"
		case archive.PhaseVerifying:
			stage = "Проверка"
		}

		parts := make([]string, 0, 4)
		if p.TotalBytes > 0 {
			parts = append(parts, humanBytes(p.ProcessedBytes)+" из "+humanBytes(p.TotalBytes))
		}
		elapsed := time.Since(started).Seconds()
		if elapsed >= 0.4 && p.ProcessedBytes > 0 {
			speed := int64(float64(p.ProcessedBytes) / elapsed)
			parts = append(parts, humanBytes(speed)+"/с")
		}
		if p.ProcessedBytes > 0 && p.StoredBytes > 0 {
			ratio := float64(p.StoredBytes) * 100 / float64(p.ProcessedBytes)
			if ratio < 999.5 {
				parts = append(parts, fmt.Sprintf("данные %.0f%%", ratio))
			}
		}
		if p.FilesDone > 0 {
			parts = append(parts, fmt.Sprintf("файлов %d", p.FilesDone))
		}
		detail := p.CurrentPath
		if len(parts) > 0 {
			status := strings.Join(parts, "  •  ")
			if detail != "" {
				detail += "  —  " + status
			} else {
				detail = status
			}
		}
		return platform.ProgressUpdate{Stage: stage, Detail: detail, Current: p.ProcessedBytes, Total: p.TotalBytes}
	}
}

func humanBytes(n int64) string {
	const unit = int64(1024)
	if n < unit {
		return fmt.Sprintf("%d Б", n)
	}
	div, exp := unit, 0
	for value := n / unit; value >= unit && exp < 5; value /= unit {
		div *= unit
		exp++
	}
	units := []string{"КиБ", "МиБ", "ГиБ", "ТиБ", "ПиБ", "ЭиБ"}
	return fmt.Sprintf("%.1f %s", float64(n)/float64(div), units[exp])
}

func topLevelInputs(root string) ([]string, error) {
	items, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, filepath.Join(root, item.Name()))
	}
	return out, nil
}

func fingerprint(root string) (string, error) {
	entries := make([]fingerprintEntry, 0, 64)
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		entries = append(entries, fingerprintEntry{Path: filepath.ToSlash(rel), Size: info.Size(), Mode: uint32(info.Mode()), MTime: info.ModTime().UnixNano()})
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	b, err := json.Marshal(entries)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func acquireLock(path string) error {
	if b, err := os.ReadFile(path); err == nil {
		pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
		if platform.ProcessRunning(pid) {
			return errors.New("этот архив уже открыт в DedupArc")
		}
		_ = os.Remove(path)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return errors.New("не удалось создать сессию архива")
	}
	_, _ = fmt.Fprintln(f, os.Getpid())
	return f.Close()
}
