//go:build !windows

package archive

import (
	"os"
	"path/filepath"
)

func walkSource(root string, visit func(path string, meta scanMeta) (bool, error)) error {
	return filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		skip, err := visit(path, scanMeta{
			IsDir:        d.IsDir(),
			Size:         info.Size(),
			Mode:         info.Mode(),
			ModifiedUnix: info.ModTime().Unix(),
			Symlink:      info.Mode()&os.ModeSymlink != 0,
		})
		if err != nil {
			return err
		}
		if skip && d.IsDir() {
			return filepath.SkipDir
		}
		return nil
	})
}
