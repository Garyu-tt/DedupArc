// Package weightbasis is an isolated, opt-in lossless compression experiment.
// It is NOT a .darc file or chunk format. Existing DedupArc archives are unaffected.
package weightbasis

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/bits"
)

const (
	MaxDecodedSize = 8 << 20
	GroupSize = 256
	maxWeightDepth = 3
	methodRaw byte = 0
	methodZlib byte = 1
	methodWeighted byte = 2
)
const magic = "WBS1"

// Candidate multipliers include non-powers of two. Zero marks a raw group.
var weights = [...]byte{2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 128, 255}
var ErrInvalid = errors.New("invalid weightbasis frame")

// Stats reports complete serialized frame sizes, including framing/metadata.
// WeightedBytes is zero if no weighted candidate was attempted.
type Stats struct {
	OriginalBytes int `json:"original_bytes"`
	RawBytes int `json:"raw_bytes"`
	ZlibBytes int `json:"zlib_bytes"`
	WeightedBytes int `json:"weighted_bytes"`
	EncodedBytes int `json:"encoded_bytes"`
	Method string `json:"method"`
	Groups int `json:"groups"`
	WeightedGroups int `json:"weighted_groups"`
}

// Encode chooses the smallest entire serialized frame among raw, zlib and
// a shared-weight quotient plus exact sparse residual model.
func Encode(input []byte) ([]byte, error) {
	frame, _, err := encode(input, 0)
	return frame, err
}

// Compare performs the same cost-based selection and returns full-size metrics.
func Compare(input []byte) (Stats, error) {
	_, stats, err := encode(input, 0)
	return stats, err
}

func encode(input []byte, depth int) ([]byte, Stats, error) {
	if len(input) > MaxDecodedSize { return nil, Stats{}, fmt.Errorf("input exceeds %d bytes", MaxDecodedSize) }
	best := frame(methodRaw, len(input), input)
	stats := Stats{OriginalBytes: len(input), RawBytes: len(best), EncodedBytes: len(best), Method: "raw"}
	var compressed bytes.Buffer
	zw, err := zlib.NewWriterLevel(&compressed, zlib.BestCompression)
	if err != nil { return nil, Stats{}, err }
	if _, err := zw.Write(input); err != nil { _ = zw.Close(); return nil, Stats{}, err }
	if err := zw.Close(); err != nil { return nil, Stats{}, err }
	zframe := frame(methodZlib, len(input), compressed.Bytes())
	stats.ZlibBytes = len(zframe)
	if len(zframe) < len(best) { best = zframe; stats.Method = "zlib" }
	if len(input) >= GroupSize && depth < maxWeightDepth {
		payload, usedGroups, err := weightedPayload(input, depth)
		if err != nil { return nil, Stats{}, err }
		candidate := frame(methodWeighted, len(input), payload)
		stats.Groups = (len(input)+GroupSize-1)/GroupSize
		stats.WeightedGroups = usedGroups
		stats.WeightedBytes = len(candidate)
		if len(candidate) < len(best) { best = candidate; stats.Method = "weighted" }
	}
	stats.EncodedBytes = len(best)
	return best, stats, nil
}

func frame(method byte, size int, payload []byte) []byte {
	b := make([]byte, 0, 5+binary.MaxVarintLen64+len(payload))
	b = append(b, magic...)
	b = append(b, method)
	b = binary.AppendUvarint(b, uint64(size))
	return append(b, payload...)
}

func bitsFor(weight byte) int { return bits.Len8(255 / weight) }
func varintLen(n uint64) int {
	var b [binary.MaxVarintLen64]byte
	return binary.PutUvarint(b[:], n)
}
func weightAllowed(w byte) bool {
	for _, valid := range weights { if w == valid { return true } }
	return false
}

func weightedPayload(input []byte, depth int) ([]byte, int, error) {
	count := (len(input)+GroupSize-1)/GroupSize
	codes := make([]byte, count)
	groups := make([][]byte, count)
	used := 0
	for i := 0; i < count; i++ {
		lo := i*GroupSize
		hi := lo+GroupSize
		if hi > len(input) { hi = len(input) }
		group := input[lo:hi]
		bestSize, bestWeight := len(group), byte(0)
		for _, w := range weights {
			nBits := bitsFor(w)
			cost := (len(group)*nBits+7)/8 + 1
			prev := -1
			patches := 0
			for p, value := range group {
				if value%w != 0 {
					cost += varintLen(uint64(p-prev)) + 1
					prev = p
					patches++
				}
			}
			cost += varintLen(uint64(patches)) - 1
			if cost < bestSize { bestSize, bestWeight = cost, w }
		}
		codes[i] = bestWeight
		if bestWeight == 0 { groups[i] = group; continue }
		used++
		w := bestWeight
		nBits := bitsFor(w)
		packed := make([]byte, (len(group)*nBits+7)/8)
		patches := make([]int, 0)
		for j, value := range group {
			if value%w != 0 { patches = append(patches, j); continue }
			quotient := value/w
			for bit := 0; bit < nBits; bit++ {
				if quotient&(1<<bit) != 0 {
					position := j*nBits+bit
					packed[position/8] |= 1<<uint(position%8)
				}
			}
		}
		encoded := binary.AppendUvarint(packed, uint64(len(patches)))
		prev := -1
		for _, at := range patches {
			encoded = binary.AppendUvarint(encoded, uint64(at-prev))
			encoded = append(encoded, group[at])
			prev = at
		}
		groups[i] = encoded
	}
	nested, _, err := encode(codes, depth+1)
	if err != nil { return nil, 0, err }
	payload := make([]byte, 0, len(nested)+varintLen(uint64(len(nested)))+len(input))
	payload = binary.AppendUvarint(payload, uint64(len(nested)))
	payload = append(payload, nested...)
	for _, group := range groups { payload = append(payload, group...) }
	return payload, used, nil
}

// Decode rejects malformed sizes, unsupported methods, invalid group references,
// excess recursion, extra bytes, and partial/corrupted zlib streams.
func Decode(data []byte) ([]byte, error) { return decode(data, 0) }

func readVarint(data *[]byte) (int, error) {
	u, n := binary.Uvarint(*data)
	if n <= 0 || u > uint64(MaxDecodedSize)+1024 { return 0, ErrInvalid }
	if n != varintLen(u) { return 0, ErrInvalid }
	*data = (*data)[n:]
	return int(u), nil
}

func decode(data []byte, depth int) ([]byte, error) {
	if len(data) < 6 || string(data[:4]) != magic { return nil, ErrInvalid }
	method := data[4]
	payload := data[5:]
	n, err := readVarint(&payload)
	if err != nil || n > MaxDecodedSize { return nil, ErrInvalid }
	switch method {
	case methodRaw:
		if len(payload) != n { return nil, ErrInvalid }
		return append([]byte(nil), payload...), nil
	case methodZlib:
		reader := bytes.NewReader(payload)
		zr, err := zlib.NewReader(reader)
		if err != nil { return nil, ErrInvalid }
		output, readErr := io.ReadAll(io.LimitReader(zr, int64(n)+1))
		closeErr := zr.Close()
		if readErr != nil || closeErr != nil || len(output) != n || reader.Len() != 0 { return nil, ErrInvalid }
		return output, nil
	case methodWeighted:
		if depth >= maxWeightDepth || n < GroupSize { return nil, ErrInvalid }
		encodedLen, err := readVarint(&payload)
		if err != nil || encodedLen < 6 || encodedLen > len(payload) { return nil, ErrInvalid }
		codes, err := decode(payload[:encodedLen], depth+1)
		if err != nil { return nil, ErrInvalid }
		payload = payload[encodedLen:]
		groupCount := (n+GroupSize-1)/GroupSize
		if len(codes) != groupCount { return nil, ErrInvalid }
		output := make([]byte, n)
		for i, w := range codes {
			lo := i*GroupSize
			hi := lo+GroupSize
			if hi > n { hi = n }
			chunk := output[lo:hi]
			if w == 0 {
				if len(payload) < len(chunk) { return nil, ErrInvalid }
				copy(chunk, payload[:len(chunk)])
				payload = payload[len(chunk):]
				continue
			}
			if !weightAllowed(w) { return nil, ErrInvalid }
			width := bitsFor(w)
			packedLen := (len(chunk)*width+7)/8
			if len(payload) < packedLen { return nil, ErrInvalid }
			packed := payload[:packedLen]
			payload = payload[packedLen:]
			for j := range chunk {
				var q int
				for bit := 0; bit < width; bit++ {
					position := j*width+bit
					if packed[position/8]&(1<<uint(position%8)) != 0 { q |= 1<<bit }
				}
				if q > 255/int(w) { return nil, ErrInvalid }
				chunk[j] = byte(q*int(w))
			}
			patchCount, err := readVarint(&payload)
			if err != nil || patchCount > len(chunk) { return nil, ErrInvalid }
			prev := -1
			for p := 0; p < patchCount; p++ {
				delta, err := readVarint(&payload)
				if err != nil || delta <= 0 || delta > len(chunk) || prev+delta >= len(chunk) || len(payload) == 0 { return nil, ErrInvalid }
				at := prev+delta
				value := payload[0]
				payload = payload[1:]
				if chunk[at] != 0 || value%w == 0 { return nil, ErrInvalid }
				chunk[at] = value
				prev = at
			}
		}
		if len(payload) != 0 { return nil, ErrInvalid }
		return output, nil
	default:
		return nil, ErrInvalid
	}
}
