package archive

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// evenKeys looks for identical upper 14 bits in aligned little-endian uint16
// words. The low two bits of every word are intentionally ignored in the
// fingerprint only: the complete block is still checked before encoding.
func evenKeys(data []byte) []deltaKey {
	if len(data) < deltaMinSize || len(data) > deltaMaxSize || len(data)%2 != 0 {
		return nil
	}
	keys := make([]deltaKey, 0, deltaSamples)
	for slot := 0; slot < deltaSamples; slot++ {
		offset := (len(data) - deltaWindow) * slot / (deltaSamples - 1)
		var sample [deltaWindow]byte
		copy(sample[:], data[offset:offset+deltaWindow])
		for i := range sample {
			if (offset+i)%2 == 0 {
				sample[i] &^= 3
			}
		}
		h := sha256.Sum256(sample[:])
		keys = append(keys, deltaKey{
			size: uint32(len(data)),
			slot: uint8(slot),
			kind: 1,
			hash: binary.LittleEndian.Uint64(h[:8]),
		})
	}
	return keys
}

func sameEvenHighBits(a, b []byte) bool {
	if len(a) != len(b) || len(a)%2 != 0 {
		return false
	}
	for i := 0; i < len(a); i += 2 {
		if a[i]&^byte(3) != b[i]&^byte(3) || a[i+1] != b[i+1] {
			return false
		}
	}
	return true
}

// Four two-bit residues per byte. Word order and byte order are fixed.
func packEvenResidues(data []byte) []byte {
	words := len(data) / 2
	out := make([]byte, (words+3)/4)
	for i := 0; i < words; i++ {
		out[i/4] |= (data[i*2] & 3) << uint((i%4)*2)
	}
	return out
}

func unpackEvenResidues(base, packed []byte) ([]byte, error) {
	if len(base)%2 != 0 || len(packed) != (len(base)/2+3)/4 {
		return nil, fmt.Errorf("неверная длина остатков even2")
	}
	words := len(base)/2
	if words%4 != 0 && len(packed) > 0 {
		if packed[len(packed)-1]>>uint((words%4)*2) != 0 {
			return nil, fmt.Errorf("ненулевые биты выравнивания even2")
		}
	}
	out := append([]byte(nil), base...)
	for i := 0; i < words; i++ {
		out[i*2] = (out[i*2] &^ byte(3)) | ((packed[i/4] >> uint((i%4)*2)) & 3)
	}
	return out, nil
}

// tryEven stores a direct base SHA-256 plus a packed two-bit residual stream.
// It never records approximate data, and only references previously written
// raw/zlib blocks. Encoding is used only when it beats existing codecs by 64B.
func (m *deltaMatcher) tryEven(data []byte, baselineSize int) (byte, []byte, bool) {
	keys := evenKeys(data)
	if len(keys) == 0 || 32+(len(data)/2+3)/4+64 >= baselineSize {
		return 0, nil, false
	}
	seen := make(map[[32]byte]struct{}, deltaMaxTry)
	for _, key := range keys {
		for i := len(m.index[key])-1; i >= 0; i-- {
			id := m.index[key][i]
			if _, exists := seen[id]; exists {
				continue
			}
			seen[id] = struct{}{}
			base, exists := m.cache[id]
			if exists && sameEvenHighBits(base, data) {
				encoded := make([]byte, 0, 32+(len(data)/2+3)/4)
				encoded = append(encoded, id[:]...)
				encoded = append(encoded, packEvenResidues(data)...)
				return methodEvenResidual, encoded, true
			}
			if len(seen) >= deltaMaxTry {
				break
			}
		}
		if len(seen) >= deltaMaxTry {
			break
		}
	}
	return 0, nil, false
}
