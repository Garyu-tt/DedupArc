package archive

import (
	"encoding/binary"
	"fmt"
)

// R03 stores a shallow, backward-only reference to a previously written
// mathematical chunk plus exact sparse replacements. The referenced chunk
// must use methodMath (not another reference or delta), so decode depth is 1.
const (
	methodMathRef byte = 4
	mathRefBudget = 32 * 1024 * 1024
	mathRefMaxRoots = 64
	mathRefMaxTry = 16
	mathRefSample = 256
)

type mathRoot struct {
	ordinal uint64
	data []byte
}
type mathRefEncoder struct {
	roots []mathRoot
	bytes int
}

func (m *mathRefEncoder) offer(ordinal uint64, data []byte) {
	if len(data) < 64 || len(data) > mathMaxRaw || len(data) > mathRefBudget {
		return
	}
	for (len(m.roots) >= mathRefMaxRoots || m.bytes + len(data) > mathRefBudget) && len(m.roots) > 0 {
		m.bytes -= len(m.roots[0].data)
		m.roots[0] = mathRoot{}
		m.roots = m.roots[1:]
	}
	m.roots = append(m.roots, mathRoot{ordinal: ordinal, data: append([]byte(nil), data...)})
	m.bytes += len(data)
}

// try compares complete encoded lengths, using a capped sample only to
// eliminate obviously unsuitable bases. No approximate recovery is possible.
func (m *mathRefEncoder) try(data []byte, baseline int) ([]byte, bool) {
	if len(data) < 64 || len(data) > mathMaxRaw || baseline < 20 {
		return nil, false
	}
	bestSize := baseline
	var best []byte
	tried := 0
	for j := len(m.roots)-1; j >= 0 && tried < mathRefMaxTry; j-- {
		root := m.roots[j]
		if len(root.data) != len(data) {
			continue
		}
		tried++
		misses := 0
		for k := 0; k < mathRefSample; k++ {
			pos := k * (len(data)-1) / (mathRefSample-1)
			if root.data[pos] != data[pos] {
				misses++
			}
		}
		if misses > mathRefSample/8 {
			continue
		}
		head := binary.AppendUvarint(nil, root.ordinal)
		patches := make([]byte, 0, 64)
		var count uint64
		prev := 0
		invalid := false
		for i, b := range data {
			if root.data[i] == b {
				continue
			}
			patches = binary.AppendUvarint(patches, uint64(i-prev))
			patches = append(patches, b)
			prev = i + 1
			count++
			// This is only an early rejection. The exact encoded size,
			// including the patch count, is tested below.
			if len(head)+len(patches)+10 >= bestSize {
				invalid = true
				break
			}
		}
		if invalid {
			continue
		}
		code := binary.AppendUvarint(head, count)
		code = append(code, patches...)
		// Require a small but real benefit to avoid needless references.
		if len(code)+16 < bestSize {
			best, bestSize = code, len(code)
		}
	}
	return best, best != nil
}

func mathRefDecode(code []byte, rawSize uint32, resolve func(uint64) ([]byte, error)) ([]byte, error) {
	if rawSize == 0 || rawSize > mathMaxRaw || len(code) < 2 {
		return nil, fmt.Errorf("math reference: invalid size")
	}
	index, n := binary.Uvarint(code)
	if n <= 0 {
		return nil, fmt.Errorf("math reference: invalid ordinal")
	}
	cursor := n
	count, n := binary.Uvarint(code[cursor:])
	if n <= 0 || count > uint64(rawSize) {
		return nil, fmt.Errorf("math reference: invalid patch count")
	}
	cursor += n
	base, err := resolve(index)
	if err != nil {
		return nil, err
	}
	if len(base) != int(rawSize) {
		return nil, fmt.Errorf("math reference: base size mismatch")
	}
	dst := append([]byte(nil), base...)
	var prev uint64
	for i := uint64(0); i < count; i++ {
		if cursor >= len(code) {
			return nil, fmt.Errorf("math reference: truncated patch")
		}
		gap, n := binary.Uvarint(code[cursor:])
		if n <= 0 {
			return nil, fmt.Errorf("math reference: invalid patch gap")
		}
		cursor += n
		if gap >= uint64(rawSize) || prev > uint64(rawSize)-gap || prev+gap >= uint64(rawSize) || cursor >= len(code) {
			return nil, fmt.Errorf("math reference: patch out of bounds")
		}
		pos := prev + gap
		dst[pos] = code[cursor]
		cursor++
		prev = pos + 1
	}
	if cursor != len(code) {
		return nil, fmt.Errorf("math reference: trailing bytes")
	}
	return dst, nil
}
