package archive

import (
	"bytes"
	"crypto/sha256"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/Garyu-tt/deduparc/internal/chunker"
)

func testEvenPair() ([]byte, []byte) {
	rng := rand.New(rand.NewSource(20261121))
	base := make([]byte, 64*1024)
	_, _ = rng.Read(base)
	changed := bytes.Clone(base)
	for i := 0; i < len(changed); i += 2 {
		changed[i] = (changed[i] &^ byte(3)) | ((changed[i] + 1) & 3)
	}
	return base, changed
}

func TestEvenResidueCodecExactRoundtrip(t *testing.T) {
	base, changed := testEvenPair()
	if !sameEvenHighBits(base, changed) {
		t.Fatal("high-bit match not detected")
	}
	residue := packEvenResidues(changed)
	if len(residue) != len(changed)/8 {
		t.Fatal("unexpected residue size")
	}
	restored, err := unpackEvenResidues(base, residue)
	if err != nil || !bytes.Equal(restored, changed) {
		t.Fatalf("roundtrip failed: %v", err)
	}
	if !bytes.Equal(base, bytes.Clone(base)) {
		t.Fatal("base mutated")
	}
	base[100] ^= 4
	if sameEvenHighBits(base, changed) {
		t.Fatal("false match when a high bit changed")
	}
}

func TestEvenResidueMalformedPayload(t *testing.T) {
	base := make([]byte, 10) // five uint16 values
	payload := packEvenResidues(base)
	if _, err := unpackEvenResidues(base, payload[:1]); err == nil {
		t.Fatal("accepted truncated residual")
	}
	payload[1] = 0xfc // non-zero unused padding bits
	if _, err := unpackEvenResidues(base, payload); err == nil {
		t.Fatal("accepted non-zero padding")
	}
	if _, err := unpackEvenResidues(base[:9], payload); err == nil {
		t.Fatal("accepted odd-sized base")
	}
}

func TestExtremeEvenMatcherRejectsNonMatchingHighBits(t *testing.T) {
	base, changed := testEvenPair()
	m := newDeltaMatcher()
	m.offer(sha256.Sum256(base), base)
	method, payload, ok := m.tryEven(changed, len(changed))
	if !ok || method != methodEvenResidual || len(payload) != 32+len(changed)/8 {
		t.Fatal("matching base did not produce even2 residual")
	}
	changed[1234] ^= 4 // outside sampled windows; full comparison must reject
	if _, _, ok := m.tryEven(changed, len(changed)); ok {
		t.Fatal("accepted an inexact high-bit match")
	}
	if _, _, ok := m.tryEven(base, 15); ok {
		t.Fatal("accepted a candidate larger than baseline")
	}
}

func TestExtremeEvenArchiveRoundtripAndSize(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.Mkdir(source, 0700); err != nil {
		t.Fatal(err)
	}
	a, b := testEvenPair()
	if err := os.WriteFile(filepath.Join(source, "a.bin"), a, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "b.bin"), b, 0600); err != nil {
		t.Fatal(err)
	}
	cfg := chunker.Config{Min: 64 * 1024, Avg: 64 * 1024, Max: 64 * 1024}
	options := CreateOptions{Compression: CompressionExtremeEven, Chunker: cfg, SmallFileThreshold: 1}
	newPath := filepath.Join(root, "even.darc")
	result, err := Create(newPath, []string{source}, options)
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.Version != 4 || result.Manifest.Stats.EvenResidualChunks == 0 {
		t.Fatalf("expected even2 in v4: version=%d stats=%+v", result.Manifest.Version, result.Manifest.Stats)
	}
	oldPath := filepath.Join(root, "maximum.darc")
	options.Compression = CompressionMaximum
	oldResult, err := Create(oldPath, []string{source}, options)
	if err != nil {
		t.Fatal(err)
	}
	if result.Bytes >= oldResult.Bytes*8/10 {
		t.Fatalf("even2 did not improve size sufficiently: %d versus %d", result.Bytes, oldResult.Bytes)
	}
	r, err := Open(newPath)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Verify(); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(root, "restore")
	if err := r.Extract(dest); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string][]byte{"a.bin": a, "b.bin": b} {
		got, err := os.ReadFile(filepath.Join(dest, "source", name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("byte mismatch after restore: %s", name)
		}
	}
}

func TestExtremeEvenCorruptReferenceIsRejected(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.Mkdir(source, 0700); err != nil {
		t.Fatal(err)
	}
	a, b := testEvenPair()
	for name, data := range map[string][]byte{"a.bin": a, "b.bin": b} {
		if err := os.WriteFile(filepath.Join(source, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(root, "corrupt.darc")
	_, err := Create(path, []string{source}, CreateOptions{
		Compression: CompressionExtremeEven,
		Chunker: chunker.Config{Min: 64 * 1024, Avg: 64 * 1024, Max: 64 * 1024},
		SmallFileThreshold: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var at int64 = -1
	for _, rec := range r.chunks {
		if rec.Method == methodEvenResidual {
			at = rec.Offset
			break
		}
	}
	_ = r.Close()
	if at < 0 {
		t.Fatal("no even2 record")
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt(bytes.Repeat([]byte{0xff}, 32), at); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	r, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Verify(); err == nil {
		t.Fatal("corrupted even2 base reference was accepted")
	}
}
