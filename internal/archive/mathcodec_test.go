package archive

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestMathCodecRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cases := map[string][]byte{
		"constant": bytes.Repeat([]byte{0x7d}, 65536),
		"affine": make([]byte, 65536),
		"periodic": make([]byte, 65536),
		"affine-patched": make([]byte, 65536),
		"random": make([]byte, 65536),
	}
	for i := 0; i < 65536; i++ {
		cases["affine"][i] = byte(11 + i*7)
		cases["affine-patched"][i] = byte(11 + i*7)
		cases["periodic"][i] = []byte("abcdefg")[i%7]
	}
	for i := 0; i < 100; i++ {
		idx := rng.Intn(65536)
		cases["affine-patched"][idx] ^= 0xff
	}
	rng.Read(cases["random"])
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			code, ok := mathTry(data, len(data))
			if name == "random" {
				if ok { t.Fatal("random data unexpectedly encoded") }
				return
			}
			if !ok { t.Fatalf("math codec failed to encode %s", name) }
			decoded, err := mathDecode(code, uint32(len(data)))
			if err != nil || !bytes.Equal(decoded, data) { t.Fatalf("roundtrip error: %v", err) }
			t.Logf("%s: %d -> %d bytes", name, len(data), len(code))
		})
	}
}

func TestMathCodecCorruption(t *testing.T) {
	code, ok := mathTry(bytes.Repeat([]byte{1}, 4096), 4096)
	if !ok { t.Fatal("not encoded") }
	for _, bad := range [][]byte{{}, {0xff, 1, 0}, {mathPeriodic, 0, 0}, {mathAffine, 2, 3, 1, 0xff}, append(bytes.Clone(code), 0)} {
		if _, err := mathDecode(bad, 4096); err == nil { t.Fatalf("accepted corruption %x", bad) }
	}
}

func TestNeutronArchiveRoundTrip(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "affine.bin")
	data := make([]byte, 256*1024)
	for i := range data { data[i] = byte(13 + i*7) }
	if err := os.WriteFile(source, data, 0600); err != nil { t.Fatal(err) }
	path := filepath.Join(dir, "neutron.darc")
	res, err := Create(path, []string{source}, CreateOptions{Compression: CompressionNeutron})
	if err != nil { t.Fatal(err) }
	if res.Manifest.Version != 4 { t.Fatalf("format version=%d, want 4", res.Manifest.Version) }
	r, err := Open(path)
	if err != nil { t.Fatal(err) }
	defer r.Close()
	foundMath := false
	for _, record := range r.chunks { if record.Method == methodMath { foundMath = true } }
	if !foundMath { t.Fatal("expected at least one math-bytecode block") }
	if _, err := r.Verify(); err != nil { t.Fatal(err) }
	out := filepath.Join(dir, "restored")
	if err := r.Extract(out); err != nil { t.Fatal(err) }
	got, err := os.ReadFile(filepath.Join(out, filepath.Base(source)))
	if err != nil { t.Fatal(err) }
	if !bytes.Equal(data, got) { t.Fatal("extracted bytes differ from original") }
}

func FuzzMathCodec(f *testing.F) {
	f.Add([]byte("abababababababababababababababab"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 64 || len(data) > 16384 { return }
		packed, ok := mathTry(data, len(data))
		if !ok { return }
		out, err := mathDecode(packed, uint32(len(data)))
		if err != nil || !bytes.Equal(out, data) { t.Fatalf("roundtrip error: %v", err) }
	})
}
