package archive

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func testVersionedBinaryFiles(t *testing.T, root string) [][]byte {
	t.Helper()
	rng := rand.New(rand.NewSource(20261108))
	base := make([]byte, 2*1024*1024)
	if _, err := rng.Read(base); err != nil {
		t.Fatal(err)
	}
	files := make([][]byte, 6)
	for n := range files {
		data := bytes.Clone(base)
		if n > 0 {
			// Real edits inside otherwise high-entropy binary files.
			for i := 0; i < 85; i++ {
				pos := rng.Intn(len(data))
				data[pos] ^= byte(n + i%250 + 1)
			}
		}
		files[n] = data
		name := filepath.Join(root, "build"+string(rune('A'+n))+".bin")
		if err := os.WriteFile(name, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return files
}

func TestExtremeDeltaRoundtripAndSize(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.Mkdir(source, 0700); err != nil {
		t.Fatal(err)
	}
	want := testVersionedBinaryFiles(t, source)
	oldPath := filepath.Join(root, "maximum.darc")
	old, err := Create(oldPath, []string{source}, CreateOptions{Compression: CompressionMaximum})
	if err != nil {
		t.Fatal(err)
	}
	if old.Manifest.Version != 2 {
		t.Fatalf("maximum unexpectedly changed format: %d", old.Manifest.Version)
	}
	newPath := filepath.Join(root, "extreme.darc")
	result, err := Create(newPath, []string{source}, CreateOptions{Compression: CompressionExtreme})
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.Version != 3 {
		t.Fatalf("want v3, got %d", result.Manifest.Version)
	}
	if result.Manifest.Stats.DeltaChunks == 0 {
		t.Fatal("expected nonzero cross-version delta blocks")
	}
	if result.Bytes >= old.Bytes*3/4 {
		t.Fatalf("delta insufficient: extreme=%d baseline=%d", result.Bytes, old.Bytes)
	}
	reader, err := Open(newPath)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if _, err := reader.Verify(); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(root, "restored")
	if err := reader.Extract(dest); err != nil {
		t.Fatal(err)
	}
	for i, data := range want {
		path := filepath.Join(dest, "source", "build"+string(rune('A'+i))+".bin")
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, data) {
			t.Fatalf("roundtrip mismatch for %d", i)
		}
	}
}

func TestExtremeRejectsBrokenDeltaBase(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.Mkdir(source, 0700); err != nil {
		t.Fatal(err)
	}
	testVersionedBinaryFiles(t, source)
	path := filepath.Join(root, "extreme.darc")
	if _, err := Create(path, []string{source}, CreateOptions{Compression: CompressionExtreme}); err != nil {
		t.Fatal(err)
	}
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var at int64 = -1
	for _, record := range r.chunks {
		if record.Method == methodDeltaZlib {
			at = record.Offset
			break
		}
	}
	_ = r.Close()
	if at < 0 {
		t.Fatal("no delta chunk found")
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt(bytes.Repeat([]byte{0xff}, 32), at); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	corrupt, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer corrupt.Close()
	if _, err := corrupt.Verify(); err == nil {
		t.Fatal("missing delta base accepted")
	}
}
