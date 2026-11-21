package weightbasis

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"testing"
)

func seeded(n int, seed int64) []byte {
	b := make([]byte, n)
	_, _ = rand.New(rand.NewSource(seed)).Read(b)
	return b
}

func TestRoundtripCorpus(t *testing.T) {
	random := seeded(64*1024, 17)
	multiples := seeded(64*1024, 19)
	for i := range multiples { multiples[i] = (multiples[i] & 15) * 16 }
	mixed := append([]byte(nil), multiples...)
	for i := 0; i < len(mixed); i += 73 { mixed[i]++ }
	heterogeneous := make([]byte, 64*1024)
	scales := []byte{2, 3, 4, 8, 16, 32, 128, 255}
	for i := range heterogeneous {
		w := scales[(i/GroupSize)%len(scales)]
		heterogeneous[i] = byte(int(random[i])%((255/int(w))+1)) * w
	}
	corpus := map[string][]byte{
		"empty": nil,
		"one": {0x3a},
		"zeros": make([]byte, 32768),
		"random": random,
		"multiples": multiples,
		"residuals": mixed,
		"heterogeneous": heterogeneous,
		"short": seeded(510, 99),
		"partial": seeded(257, 100),
		"max": make([]byte, MaxDecodedSize),
	}
	for name, original := range corpus {
		t.Run(name, func(t *testing.T) {
			coded, err := Encode(original)
			if err != nil { t.Fatal(err) }
			restored, err := Decode(coded)
			if err != nil { t.Fatal(err) }
			if !bytes.Equal(original, restored) { t.Fatal("not byte-exact") }
			stats, err := Compare(original)
			if err != nil { t.Fatal(err) }
			if stats.EncodedBytes != len(coded) { t.Fatalf("wrong size stats: %v", stats) }
			if stats.EncodedBytes > stats.RawBytes || stats.EncodedBytes > stats.ZlibBytes { t.Fatalf("bad size selection: %v", stats) }
			if stats.WeightedBytes != 0 && stats.EncodedBytes > stats.WeightedBytes { t.Fatalf("weighted missed: %v", stats) }
		})
	}
}

func TestWeightedSelectedOnRandomQuotients(t *testing.T) {
	input := seeded(16*1024, 42)
	for i := range input { input[i] = (input[i] & 15) << 4 }
	coded, err := Encode(input)
	if err != nil { t.Fatal(err) }
	stats, err := Compare(input)
	if err != nil { t.Fatal(err) }
	if stats.Method != "weighted" || stats.WeightedGroups == 0 {
		t.Fatalf("expected weight factoring to win: %+v", stats)
	}
	restored, err := Decode(coded)
	if err != nil || !bytes.Equal(input, restored) { t.Fatalf("roundtrip: %v", err) }
}

func TestInvalidFrames(t *testing.T) {
	valid, err := Encode(seeded(2048, 301))
	if err != nil { t.Fatal(err) }
	truncated := append([]byte(nil), valid[:len(valid)-1]...)
	for name, data := range map[string][]byte{
		"missing": nil,
		"magic": []byte("BAD!!\x00"),
		"unknown-method": []byte("WBS1\x03\x00"),
		"overflow": append([]byte("WBS1\x00"), bytes.Repeat([]byte{0xff}, 11)...),
		"noncanonical-varint": []byte("WBS1\x00\x80\x00"),
		"large-size": frame(methodRaw, MaxDecodedSize+1, nil),
		"short": truncated,
		"trailing": append(valid, 0xde),
		"fake-weight": frame(methodWeighted, 256, []byte{0}),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Decode(data); err == nil { t.Fatal("accepted invalid frame") }
		})
	}
}

func TestInvalidWeightedGroup(t *testing.T) {
	// One 256-byte group with weight=3 and no patches, except invalid q=127.
	child := frame(methodRaw, 1, []byte{3})
	payload := binary.AppendUvarint(nil, uint64(len(child)))
	payload = append(payload, child...)
	q := make([]byte, 224)
	q[0] = 127 // max allowed q is 85
	payload = append(payload, q...)
	payload = append(payload, 0)
	if _, err := Decode(frame(methodWeighted, 256, payload)); err == nil { t.Fatal("accepted overflowing quotient") }
	q[0] = 0
	payload = binary.AppendUvarint(nil, uint64(len(child)))
	payload = append(payload, child...)
	payload = append(payload, q...)
	payload = append(payload, 1, 0, 0x01)
	if _, err := Decode(frame(methodWeighted, 256, payload)); err == nil { t.Fatal("accepted zero delta") }
}

func TestInputLimit(t *testing.T) {
	if _, err := Encode(make([]byte, MaxDecodedSize+1)); err == nil { t.Fatal("oversized input accepted") }
}

func FuzzDecode(f *testing.F) {
	for _, b := range [][]byte{nil, []byte("WBS1\x00\x00"), frame(methodRaw, 3, []byte{1, 2, 3})} { f.Add(b) }
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 64*1024 { t.Skip() }
		_, _ = Decode(data)
	})
}
