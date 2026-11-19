package archive

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/Garyu-tt/deduparc/internal/chunker"
)

func TestNeutronDAGCrossFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "source")
	if err := os.Mkdir(source, 0755); err != nil { t.Fatal(err) }

	const size = 512 * 1024
	pattern := []byte("deduparc-neutron-dag-period-31!")
	root := make([]byte, size)
	for i := range root { root[i] = pattern[i%len(pattern)] }
	expected := make(map[string][]byte)
	for n := 0; n < 8; n++ {
		data := bytes.Clone(root)
		if n > 0 {
			for j := 0; j < n * 2; j++ {
				pos := 1024 + j*17011
				data[pos] ^= byte(11+n+j)
			}
		}
		name := fmt.Sprintf("%02d-pattern.bin", n)
		expected[name] = data
		if err := os.WriteFile(filepath.Join(source, name), data, 0600); err != nil { t.Fatal(err) }
	}

	opts := CreateOptions{Compression: CompressionNeutronDAG, Chunker: chunker.Config{
		Min: size, Avg: size, Max: size,
	}}
	out := filepath.Join(dir, "dag.darc")
	result, err := Create(out, []string{source}, opts)
	if err != nil { t.Fatal(err) }
	if result.Manifest.Version != 5 { t.Fatalf("version=%d, expected 5", result.Manifest.Version) }
	if result.Manifest.Stats.MathChunks == 0 || result.Manifest.Stats.MathRefChunks == 0 {
		t.Fatalf("expected math roots and refs, got %+v", result.Manifest.Stats)
	}
	r, err := Open(out)
	if err != nil { t.Fatal(err) }
	defer r.Close()
	for _, record := range r.chunks {
		if record.Method != methodMathRef { continue }
		idx := record.Ordinal
		packed := make([]byte, record.PackedSize)
		if _, err := r.file.ReadAt(packed, record.Offset); err != nil { t.Fatal(err) }
		base, n := binary.Uvarint(packed)
		if n <= 0 || base >= idx || r.chunks[r.chunkOrder[base]].Method != methodMath {
			t.Fatalf("invalid forward/nested ref at ordinal %d", idx)
		}
	}
	if _, err := r.Verify(); err != nil { t.Fatal(err) }
	restored := filepath.Join(dir, "restored")
	if err := r.Extract(restored); err != nil { t.Fatal(err) }
	for name, data := range expected {
		got, err := os.ReadFile(filepath.Join(restored, "source", name))
		if err != nil { t.Fatal(err) }
		if !bytes.Equal(got, data) { t.Fatalf("%s mismatch", name) }
	}

	r02, err := Create(filepath.Join(dir, "r02.darc"), []string{source},
		CreateOptions{Compression: CompressionNeutron, Chunker: opts.Chunker})
	if err != nil { t.Fatal(err) }
	if r02.Manifest.Version != 4 { t.Fatalf("R02 version changed: %d", r02.Manifest.Version) }
	if result.Bytes >= r02.Bytes {
		t.Fatalf("DAG failed to improve deterministic cross-file corpus: R03=%d R02=%d", result.Bytes, r02.Bytes)
	}
	t.Logf("source=%d R02=%d R03=%d math-refs=%d", size*8, r02.Bytes, result.Bytes, result.Manifest.Stats.MathRefChunks)
}

func TestMathReferenceDecodeRejectsMalformed(t *testing.T) {
	base := bytes.Repeat([]byte("ABCDEF"), 100)
	resolve := func(index uint64) ([]byte,error) {
		if index != 0 {return nil, fmt.Errorf("invalid root")}
		return base, nil
	}
	good := []byte{0, 1, 3, 'Q'} // ordinal 0, 1 patch, gap 3, replacement Q
	got, err := mathRefDecode(good, uint32(len(base)), resolve)
	if err != nil || got[3]!='Q' || !bytes.Equal(got[4:], base[4:]) {
		t.Fatalf("valid reference rejected: %v",err)
	}
	bad := [][]byte{
		nil, {0x80}, {0, 0x80}, {0, 0}, // final entry with no patches is valid!
		{0, 1, 0x80}, {0, 1, 0x83, 0x7f, 4}, // out of bounds
		{0, 1, 3}, {0, 1, 3, 'Q', 0},
		{1, 0}, {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0},
	}
	// Empty-patch streams are valid, so do not treat {0,0} as corruption.
	bad = append(bad[:3], bad[4:]...)
	for _, code := range bad {
		if _, err := mathRefDecode(code, uint32(len(base)), resolve); err == nil {
			t.Fatalf("accepted malformed reference %x", code)
		}
	}
}
