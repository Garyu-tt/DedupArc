package archive

import (
	"crypto/sha256"
	"encoding/binary"
)

// basisKeys fingerprints only the high byte of each little-endian uint16.
// This permits matching blocks with dense changes in their low bytes, where
// there are no identical 64-byte windows and ordinary XOR-delta discovery
// fails. These hashes are only approximate candidate lookup, NEVER a
// correctness decision. The eventual XOR delta is exact and hash-verified.
func basisKeys(data []byte) []deltaKey {
	if len(data) < deltaMinSize || len(data) > deltaMaxSize || len(data)%2 != 0 {
		return nil
	}
	keys := make([]deltaKey, 0, deltaSamples)
	for slot := 0; slot < deltaSamples; slot++ {
		offset := (len(data)-deltaWindow)*slot/(deltaSamples-1)
		var sample [deltaWindow]byte
		copy(sample[:], data[offset:offset+deltaWindow])
		for i := range sample {
			if (offset+i)%2 == 0 {
				sample[i] = 0
			}
		}
		digest := sha256.Sum256(sample[:])
		keys = append(keys, deltaKey{
			size: uint32(len(data)),
			slot: uint8(slot),
			kind: 2,
			hash: binary.LittleEndian.Uint64(digest[:8]),
		})
	}
	return keys
}

// tryBasis reuses the existing v3 XOR-delta encoding, but tests candidates
// with a shared high-byte basis rather than requiring identical raw samples.
// The same direct (non-delta) reference can be reused by many target chunks.
// Always choose by exact physical size, including the 32-byte base reference.
func (m *deltaMatcher) tryBasis(data []byte, baselineSize int) (byte, []byte, bool, error) {
	keys := basisKeys(data)
	if len(keys) == 0 || baselineSize < 64 {
		return 0, nil, false, nil
	}
	seen := make(map[[32]byte]struct{}, deltaMaxTry)
	var best []byte
	bestSize := baselineSize
	for _, key := range keys {
		for i := len(m.index[key])-1; i >= 0; i-- {
			id := m.index[key][i]
			if _, ok := seen[id]; ok {
				continue
			}
			if len(seen) >= deltaMaxTry {
				return chooseBasisDelta(best)
			}
			seen[id] = struct{}{}
			base, ok := m.cache[id]
			if !ok || len(base) != len(data) {
				continue
			}
			delta := make([]byte, len(data))
			for i, b := range data {
				delta[i] = b ^ base[i]
			}
			zipped, err := zlibCompress(delta, 9)
			if err != nil {
				return 0, nil, false, err
			}
			if len(zipped)+32+64 < bestSize {
				best = make([]byte, 32, 32+len(zipped))
				copy(best, id[:])
				best = append(best, zipped...)
				bestSize = len(best)
			}
		}
	}
	return chooseBasisDelta(best)
}

func chooseBasisDelta(best []byte) (byte, []byte, bool, error) {
	if len(best) == 0 {
		return 0, nil, false, nil
	}
	return methodDeltaZlib, best, true, nil
}
