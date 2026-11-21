// weightbench is a repeatable, read-only benchmark for the weightbasis lab codec.
// It never creates or modifies .darc archives.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/Garyu-tt/deduparc/internal/weightbasis"
)

type result struct {
	Name string `json:"name"`
	weightbasis.Stats
	VerifiedSHA256 bool `json:"verified_sha256"`
	EncodeMillis float64 `json:"encode_millis"`
	DecodeMillis float64 `json:"decode_millis"`
}

func run(name string, input []byte) (result, error) {
	start := time.Now()
	encoded, err := weightbasis.Encode(input)
	encodeTime := time.Since(start)
	if err != nil { return result{}, err }
	start = time.Now()
	decoded, err := weightbasis.Decode(encoded)
	decodeTime := time.Since(start)
	if err != nil { return result{}, err }
	verified := sha256.Sum256(decoded) == sha256.Sum256(input) && bytes.Equal(decoded, input)
	if !verified { return result{}, fmt.Errorf("roundtrip mismatch for %s", name) }
	stats, err := weightbasis.Compare(input)
	if err != nil { return result{}, err }
	return result{Name: name, Stats: stats, VerifiedSHA256: verified,
		EncodeMillis: float64(encodeTime)/float64(time.Millisecond),
		DecodeMillis: float64(decodeTime)/float64(time.Millisecond)}, nil
}

func corpus(n int, seed int64) map[string][]byte {
	random := make([]byte, n)
	_, _ = rand.New(rand.NewSource(seed)).Read(random)
	multiples := append([]byte(nil), random...)
	for i := range multiples { multiples[i] = (multiples[i] & 15) << 4 }
	residual := append([]byte(nil), multiples...)
	for i := 0; i < n; i += 79 { residual[i]++ }
	mixed := make([]byte, n)
	scales := []byte{2, 3, 4, 8, 16, 32, 64, 128}
	for i := range mixed {
		w := scales[(i/weightbasis.GroupSize)%len(scales)]
		mixed[i] = byte(int(random[i])%((255/int(w))+1)) * w
	}
	return map[string][]byte{
		"random-control": random,
		"multiples16": multiples,
		"sparse-residuals": residual,
		"mixed-group-weights": mixed,
		"zeros-control": make([]byte, n),
	}
}

func main() {
	inputFile := flag.String("file", "", "an existing input file (up to 8 MiB); omit for deterministic lab corpus")
	size := flag.Int("size", 65536, "deterministic corpus bytes (256..8388608)")
	seed := flag.Int64("seed", 20261122, "deterministic test seed")
	jsonOutput := flag.Bool("json", false, "emit machine-readable JSON lines")
	flag.Parse()
	var samples map[string][]byte
	if *inputFile != "" {
		data, err := os.ReadFile(*inputFile)
		if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
		samples = map[string][]byte{*inputFile: data}
	} else {
		if *size < 256 || *size > weightbasis.MaxDecodedSize { fmt.Fprintln(os.Stderr, "-size must be 256..8388608"); os.Exit(2) }
		samples = corpus(*size, *seed)
	}
	names := []string{"random-control", "multiples16", "sparse-residuals", "mixed-group-weights", "zeros-control"}
	if *inputFile != "" { names = []string{*inputFile} }
	for _, name := range names {
		r, err := run(name, samples[name])
		if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
		if *jsonOutput {
			encoded, _ := json.Marshal(r)
			fmt.Println(string(encoded))
		} else {
			fmt.Printf("%-22s %8d -> %8d bytes  %-8s  raw:%8d zlib:%8d weighted:%8d  SHA256:%t encode:%.1fms decode:%.1fms
",
				r.Name, r.OriginalBytes, r.EncodedBytes, r.Method, r.RawBytes, r.ZlibBytes, r.WeightedBytes,
				r.VerifiedSHA256, r.EncodeMillis, r.DecodeMillis)
		}
	}
}
