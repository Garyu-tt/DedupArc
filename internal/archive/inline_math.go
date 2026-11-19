package archive

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
)

// R04: pure mathematical files are represented by a single shared formula
// kept in the compressed manifest, not by many physical 45-byte CHNK records.
// Related files can reference this formula with exact sparse replacements.
const (
	inlineMathMinRaw = 64 * 1024
	inlineMathMaxRaw = 8 * 1024 * 1024
	inlineMathMaxCode = 4096
	inlineMathMaxRoots = 8192
)

type InlineMathRoot struct {
	Size uint32 `json:"s"`
	Code []byte `json:"c"`
}

// A generator without patches is safe as a reusable root. All other
// mathematical blocks remain handled by the R03 physical-chunk path.
func isPureMathCode(code []byte) bool {
	if len(code) < 3 { return false }
	offset := 0
	switch code[0] {
	case mathConstant:
		offset = 2
	case mathAffine:
		offset = 3
	case mathQuadratic:
		offset = 4
	case mathPeriodic:
		if len(code) < 3 { return false }
		offset = 2 + int(code[1])
	default:
		return false
	}
	if offset >= len(code) { return false }
	count, n := binary.Uvarint(code[offset:])
	return n > 0 && count == 0 && offset+n == len(code)
}

func (w *writer) tryInlineMathFile(item sourceEntry) (bool, error) {
	if w.inlineMatcher == nil || item.size < inlineMathMinRaw ||
		item.size > inlineMathMaxRaw || isProbablyPrecompressedPath(item.storedPath) {
		return false, nil
	}
	// Separate source read is bounded at 8 MiB; ordinary CDC remains
	// available for incompressible files. Never retain the whole file.
	data, err := os.ReadFile(item.sourcePath)
	if err != nil { return false, err }
	if int64(len(data)) != item.size {
		return false, fmt.Errorf("%s: size changed during reading", item.storedPath)
	}
	if _, ok := mathTryDense(data[:min(len(data), 4096)], 256); !ok {
		return false, nil
	}

	// Reuse an existing clean mathematical root if the exact patch stream
	// is small. Root indices are logical manifest indices (not CHNK ordinals).
	refCode, hasRef := w.inlineMatcher.try(data, inlineMathMaxCode)
	mathCode, hasPure := mathTryDense(data, inlineMathMaxCode)
	hasPure = hasPure && isPureMathCode(mathCode) &&
		len(w.manifest.MathRoots) < inlineMathMaxRoots

	if !hasRef && !hasPure { return false, nil }

	var code []byte
	if hasRef {
		code = refCode
	}
	if hasPure && (!hasRef || len(mathCode)+3 < len(refCode)) {
		rootIndex := uint64(len(w.manifest.MathRoots))
		w.manifest.MathRoots = append(w.manifest.MathRoots, InlineMathRoot{
			Size: uint32(len(data)), Code: mathCode,
		})
		w.inlineMatcher.offer(rootIndex, data)
		code = binary.AppendUvarint(nil, rootIndex)
		code = binary.AppendUvarint(code, 0)
	}

	digest := sha256.Sum256(data)
	entry := Entry{
		Path: item.storedPath, Type: "file", Size: item.size,
		Mode: uint32(item.mode.Perm()), ModifiedUnix: item.modifiedUnix,
		SHA256: hex.EncodeToString(digest[:]), InlineMath: code,
	}
	w.manifest.Entries = append(w.manifest.Entries, entry)
	w.manifest.Stats.Files++
	w.manifest.Stats.Chunks++
	w.manifest.Stats.InlineMathFiles++
	w.manifest.Stats.SourceBytes += item.size
	w.processedBytes += item.size
	w.filesDone++
	if !w.notify(PhaseCompressing, item.storedPath) { return false, ErrCancelled }
	return true, nil
}

func (r *Reader) decodeInlineMath(entry Entry) ([]byte, error) {
	if entry.Size < 1 || entry.Size > inlineMathMaxRaw {
		return nil, fmt.Errorf("inline math: invalid file size")
	}
	return mathRefDecode(entry.InlineMath, uint32(entry.Size), func(index uint64) ([]byte, error) {
		if index >= uint64(len(r.Manifest.MathRoots)) {
			return nil, fmt.Errorf("inline math: missing root")
		}
		root := r.Manifest.MathRoots[index]
		if root.Size != uint32(entry.Size) { return nil, fmt.Errorf("inline math: mismatched root size") }
		return mathDecode(root.Code, root.Size)
	})
}
