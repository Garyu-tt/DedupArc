package archive

import (
	"crypto/sha256"
	"encoding/binary"
)

// Delta codec R01: exact-length CDC blocks only. A match must share at least
// one of eight 64-byte sample windows. No probabilistic correctness decisions:
// the entire XOR is stored, and SHA-256 is verified on restoration.
const (
	deltaMinSize = 64 * 1024
	deltaMaxSize = 1024 * 1024
	deltaBudget  = 64 * 1024 * 1024
	deltaMaxTry  = 4
	deltaSamples = 8
	deltaWindow  = 64
)

type deltaKey struct {
	size uint32
	slot uint8
	hash uint64
}

type deltaMatcher struct {
	index map[deltaKey][][32]byte
	cache map[[32]byte][]byte
	order [][32]byte
	head  int
	bytes int
}

func newDeltaMatcher() *deltaMatcher {
	return &deltaMatcher{
		index: make(map[deltaKey][][32]byte),
		cache: make(map[[32]byte][]byte),
	}
}

func deltaKeys(data []byte) []deltaKey {
	if len(data) < deltaMinSize || len(data) > deltaMaxSize {
		return nil
	}
	keys := make([]deltaKey, 0, deltaSamples)
	for slot := 0; slot < deltaSamples; slot++ {
		offset := (len(data) - deltaWindow) * slot / (deltaSamples - 1)
		h := sha256.Sum256(data[offset : offset+deltaWindow])
		keys = append(keys, deltaKey{
			size: uint32(len(data)), slot: uint8(slot),
			hash: binary.LittleEndian.Uint64(h[:8]),
		})
	}
	return keys
}

func (m *deltaMatcher) offer(digest [32]byte, data []byte) {
	keys := deltaKeys(data)
	if len(keys) == 0 || m.cache[digest] != nil {
		return
	}
	for m.bytes+len(data) > deltaBudget && m.head < len(m.order) {
		old := m.order[m.head]
		m.head++
		if cached, ok := m.cache[old]; ok {
			m.bytes -= len(cached)
			delete(m.cache, old)
			for _, key := range deltaKeys(cached) {
				ids := m.index[key]
				for i, id := range ids {
					if id == old {
						ids = append(ids[:i], ids[i+1:]...)
						break
					}
				}
				if len(ids) == 0 {
					delete(m.index, key)
				} else {
					m.index[key] = ids
				}
			}
		}
	}
	if m.head > 4096 && m.head*2 >= len(m.order) {
		m.order = append([][32]byte(nil), m.order[m.head:]...)
		m.head = 0
	}
	m.cache[digest] = append([]byte(nil), data...)
	m.order = append(m.order, digest)
	m.bytes += len(data)
	for _, key := range keys {
		candidates := append(m.index[key], digest)
		if len(candidates) > deltaMaxTry {
			candidates = candidates[len(candidates)-deltaMaxTry:]
		}
		m.index[key] = candidates
	}
}

// try accepts a delta only when its exact stored byte length (including the
// 32-byte reference SHA-256) substantially beats the best conventional codec.
// Bases are never themselves deltas: decoding is depth-one, with no cycles.
func (m *deltaMatcher) try(data []byte, baselineSize int) (byte, []byte, bool, error) {
	keys := deltaKeys(data)
	if len(keys) == 0 || baselineSize < len(data)/16 {
		return 0, nil, false, nil
	}
	seen := make(map[[32]byte]struct{})
	bestSize := baselineSize
	var bestPacked []byte
	for _, key := range keys {
		for i := len(m.index[key]) - 1; i >= 0; i-- {
			id := m.index[key][i]
			if _, exists := seen[id]; exists {
				continue
			}
			seen[id] = struct{}{}
			base, exists := m.cache[id]
			if !exists || len(base) != len(data) {
				continue
			}
			delta := make([]byte, len(data))
			changes := 0
			for i, b := range data {
				delta[i] = b ^ base[i]
				if delta[i] != 0 {
					changes++
				}
			}
			if changes > len(data)/6 {
				continue
			}
			zipped, err := zlibCompress(delta, 9)
			if err != nil {
				return 0, nil, false, err
			}
			if len(zipped)+32+64 < bestSize {
				bestPacked = append(append(make([]byte, 0, len(zipped)+32), id[:]...), zipped...)
				bestSize = len(bestPacked)
			}
			if len(seen) >= deltaMaxTry {
				break
			}
		}
		if len(seen) >= deltaMaxTry {
			break
		}
	}
	if len(bestPacked) == 0 {
		return 0, nil, false, nil
	}
	return methodDeltaZlib, bestPacked, true, nil
}
