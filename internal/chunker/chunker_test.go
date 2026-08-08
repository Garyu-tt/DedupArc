package chunker

import (
	"bytes"
	"crypto/sha256"
	"math/rand"
	"testing"
)

func TestSplitRoundTrip(t *testing.T) {
	data := bytes.Repeat([]byte("abcdefghijklmnopqrstuvwxyz0123456789"), 50000)
	var chunks [][]byte

	err := Split(bytes.NewReader(data), DefaultConfig(), func(chunk []byte) error {
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("ожидалось несколько чанков, получено %d", len(chunks))
	}
	if got := bytes.Join(chunks, nil); !bytes.Equal(got, data) {
		t.Fatal("данные после разбиения не совпадают с исходными")
	}
}

func TestSplitResynchronizesAfterInsertion(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := make([]byte, 4*1024*1024)
	if _, err := rng.Read(data); err != nil {
		t.Fatal(err)
	}
	modified := append([]byte("вставка в начале файла"), data...)

	originalHashes := chunkHashes(t, data)
	modifiedHashes := chunkHashes(t, modified)
	common := 0
	for hash := range originalHashes {
		if _, ok := modifiedHashes[hash]; ok {
			common++
		}
	}
	if common < len(originalHashes)/2 {
		t.Fatalf("после вставки переиспользуется слишком мало чанков: %d из %d", common, len(originalHashes))
	}
}

func chunkHashes(t *testing.T, data []byte) map[[32]byte]struct{} {
	t.Helper()
	result := make(map[[32]byte]struct{})
	if err := Split(bytes.NewReader(data), DefaultConfig(), func(chunk []byte) error {
		result[sha256.Sum256(chunk)] = struct{}{}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return result
}
