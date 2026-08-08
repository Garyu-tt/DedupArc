package archive

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
)

type VerifyResult struct {
	Files        int
	Chunks       int
	UniqueChunks int
}

func (r *Reader) Verify() (VerifyResult, error) {
	for hash := range r.chunks {
		if _, err := r.readChunk(hash); err != nil {
			return VerifyResult{}, err
		}
	}

	files := 0
	for _, entry := range r.Manifest.Entries {
		if entry.Type != "file" {
			continue
		}
		files++
		var h hash.Hash
		if entry.SHA256 != "" {
			h = sha256.New()
		}
		var size int64
		if err := r.entryData(entry, func(data []byte) error {
			if h != nil {
				_, _ = h.Write(data)
			}
			size += int64(len(data))
			return nil
		}); err != nil {
			return VerifyResult{}, fmt.Errorf("%s: %w", entry.Path, err)
		}
		if size != entry.Size {
			return VerifyResult{}, fmt.Errorf("%s: размер файла не совпадает", entry.Path)
		}
		if h != nil && hex.EncodeToString(h.Sum(nil)) != entry.SHA256 {
			return VerifyResult{}, fmt.Errorf("%s: SHA-256 файла не совпадает", entry.Path)
		}
	}
	return VerifyResult{
		Files:        files,
		Chunks:       r.Manifest.Stats.Chunks,
		UniqueChunks: len(r.chunks),
	}, nil
}
