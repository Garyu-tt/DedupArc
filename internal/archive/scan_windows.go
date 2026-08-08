//go:build windows

package archive

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

const fileAttributeReparsePoint = 0x00000400

type winFindData = syscall.Win32finddata

func walkSource(root string, visit func(path string, meta scanMeta) (bool, error)) error {
	root = filepath.Clean(root)
	info, err := os.Lstat(root)
	if err != nil {
		return err
	}
	skip, err := visit(root, scanMeta{
		IsDir:        info.IsDir(),
		Size:         info.Size(),
		Mode:         info.Mode(),
		ModifiedUnix: info.ModTime().Unix(),
		Symlink:      info.Mode()&os.ModeSymlink != 0,
	})
	if err != nil || skip || !info.IsDir() {
		return err
	}
	return walkWindowsDir(root, visit)
}

func walkWindowsDir(dir string, visit func(path string, meta scanMeta) (bool, error)) error {
	pattern, err := syscall.UTF16PtrFromString(filepath.Join(dir, "*"))
	if err != nil {
		return err
	}
	var data winFindData
	h, err := syscall.FindFirstFile(pattern, &data)
	if err != nil {
		if errors.Is(err, syscall.ERROR_FILE_NOT_FOUND) {
			return nil
		}
		return err
	}
	defer syscall.FindClose(h)

	for {
		name := syscall.UTF16ToString(data.FileName[:])
		if name != "" && name != "." && name != ".." {
			path := filepath.Join(dir, name)
			attrs := data.FileAttributes
			isDir := attrs&syscall.FILE_ATTRIBUTE_DIRECTORY != 0
			mode := os.FileMode(0o666)
			if isDir {
				mode = 0o777 | os.ModeDir
			}
			if attrs&syscall.FILE_ATTRIBUTE_READONLY != 0 {
				mode &^= 0o222
			}
			size := int64(uint64(data.FileSizeHigh)<<32 | uint64(data.FileSizeLow))
			meta := scanMeta{
				IsDir:        isDir,
				Size:         size,
				Mode:         mode,
				ModifiedUnix: time.Unix(0, data.LastWriteTime.Nanoseconds()).Unix(),
				Symlink:      attrs&fileAttributeReparsePoint != 0,
			}
			skip, visitErr := visit(path, meta)
			if visitErr != nil {
				return visitErr
			}
			if isDir && !skip {
				if err := walkWindowsDir(path, visit); err != nil {
					return err
				}
			}
		}
		if err := syscall.FindNextFile(h, &data); err != nil {
			if errors.Is(err, syscall.ERROR_NO_MORE_FILES) {
				return nil
			}
			return err
		}
	}
}
