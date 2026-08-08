package archive

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
)

// Catalog содержит только структуру архива. Для его чтения не требуется
// проходить по всем блокам данных, поэтому список файлов открывается быстро.
type Catalog struct {
	Manifest Manifest
	FileSize int64
}

func ReadCatalog(path string) (Catalog, error) {
	f, err := os.Open(path)
	if err != nil {
		return Catalog{}, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return Catalog{}, err
	}
	if info.Size() < int64(len(fileMagic)+footerSize) {
		return Catalog{}, ErrInvalidArchive
	}

	header := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(f, header); err != nil {
		return Catalog{}, err
	}
	if !bytes.Equal(header, fileMagic[:]) {
		return Catalog{}, ErrInvalidArchive
	}

	footer := make([]byte, footerSize)
	if _, err := f.ReadAt(footer, info.Size()-footerSize); err != nil {
		return Catalog{}, err
	}
	if !bytes.Equal(footer[:4], footerMagic[:]) {
		return Catalog{}, ErrInvalidArchive
	}

	manifestOffset := int64(binary.LittleEndian.Uint64(footer[4:12]))
	if manifestOffset < int64(len(fileMagic)) || manifestOffset >= info.Size()-footerSize {
		return Catalog{}, ErrInvalidArchive
	}

	manifest, err := readManifestAt(f, manifestOffset, footer[12:44])
	if err != nil {
		return Catalog{}, err
	}
	return Catalog{Manifest: manifest, FileSize: info.Size()}, nil
}
