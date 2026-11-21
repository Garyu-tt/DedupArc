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

func makeSharedBasisBlocks() [][]byte {
	const blockSize = 64 * 1024
	rng := rand.New(rand.NewSource(20261121))
	base := make([]byte, blockSize)
	for i := 0; i < len(base); i += 2 {
		base[i] = byte(rng.Intn(240))
		base[i+1] = byte(rng.Intn(256))
	}
	blocks := make([][]byte, 4)
	for n := range blocks {
		data := bytes.Clone(base)
		for i := 0; i < len(data); i += 2 {
			data[i] += byte(n)
		}
		blocks[n] = data
	}
	return blocks
}

func TestBasisKeysDenseLowByteChanges(t *testing.T) {
	blocks := makeSharedBasisBlocks()
	ka := basisKeys(blocks[0])
	kb := basisKeys(blocks[1])
	if len(ka) != deltaSamples || len(kb) != deltaSamples {
		t.Fatal("missing coarse fingerprints")
	}
	for i := range ka {
		if ka[i] != kb[i] {
			t.Fatal("shared high bytes produced different fingerprints")
		}
	}
	for _, k := range deltaKeys(blocks[0]) {
		for _, other := range deltaKeys(blocks[1]) {
			if k == other {
				t.Fatal("unexpected exact raw window: test requires dense differences")
			}
		}
	}
	if len(basisKeys(blocks[0][:len(blocks[0])-1])) != 0 {
		t.Fatal("accepted odd-length blocks")
	}
}

func TestSharedBasisMatcherExactRoundtrip(t *testing.T) {
	blocks := makeSharedBasisBlocks()
	m := newDeltaMatcher()
	m.basisEnabled = true
	baseHash := sha256.Sum256(blocks[0])
	m.offer(baseHash, blocks[0])
	for n := 1; n < len(blocks); n++ {
		method, payload, ok, err := m.tryBasis(blocks[n], len(blocks[n]))
		if err != nil || !ok || method != methodDeltaZlib {
			t.Fatalf("no delta for %d: ok=%t err=%v", n, ok, err)
		}
		if !bytes.Equal(payload[:32], baseHash[:]) {
			t.Fatal("shared base was not reused")
		}
		restoredDelta, err := unpackData(methodZlib, payload[32:], uint32(len(blocks[n])))
		if err != nil {
			t.Fatal(err)
		}
		for i := range restoredDelta {
			restoredDelta[i] ^= blocks[0][i]
		}
		if !bytes.Equal(restoredDelta, blocks[n]) {
			t.Fatalf("lossy reconstruction of block %d", n)
		}
	}
	// Never accept a delta which is bigger than the conventional encoding.
	if _, _, ok, _ := m.tryBasis(blocks[1], 100); ok {
		t.Fatal("accepted a non-beneficial delta")
	}
}

func TestSharedBasisDifferentHighBytesFallback(t *testing.T) {
	blocks := makeSharedBasisBlocks()
	m := newDeltaMatcher()
	m.basisEnabled = true
	m.offer(sha256.Sum256(blocks[0]), blocks[0])
	other := bytes.Clone(blocks[1])
	for i := 1; i < len(other); i += 2 {
		other[i] ^= 0xff
	}
	if _, _, ok, err := m.tryBasis(other, len(other)); err != nil || ok {
		t.Fatalf("unrelated high bytes unexpectedly found a basis: ok=%v err=%v", ok, err)
	}
}

func TestSharedBasisArchiveRoundtripAndRatio(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "source")
	if err := os.Mkdir(src, 0700); err != nil {
		t.Fatal(err)
	}
	blocks := makeSharedBasisBlocks()
	for i, data := range blocks {
		if err := os.WriteFile(filepath.Join(src, string(rune('a'+i))+".bin"), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	cfg := chunker.Config{Min: 64 * 1024, Avg: 64 * 1024, Max: 64 * 1024}
	options := CreateOptions{Compression: CompressionExtreme, Chunker: cfg, SmallFileThreshold: 1}
	baseline, err := Create(filepath.Join(root, "extreme.darc"), []string{src}, options)
	if err != nil {
		t.Fatal(err)
	}
	options.Compression = CompressionExtremeBasis
	path := filepath.Join(root, "basis.darc")
	result, err := Create(path, []string{src}, options)
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.Version != 3 || result.Manifest.Stats.BasisChunks < 3 {
		t.Fatalf("basis mode did not reuse shared base: %+v", result.Manifest.Stats)
	}
	if result.Bytes >= baseline.Bytes*8/10 {
		t.Fatalf("shared basis was not smaller: shared=%d extreme=%d", result.Bytes, baseline.Bytes)
	}
	reader, err := Open(path)
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
	for i, want := range blocks {
		name := string(rune('a'+i))+".bin"
		got, err := os.ReadFile(filepath.Join(dest, "source", name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("shared basis corrupted %s", name)
		}
	}
}
