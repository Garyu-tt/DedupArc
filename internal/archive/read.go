package archive

import (
	"bytes"
	"compress/zlib"
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type chunkRecord struct {
	Offset     int64
	RawSize    uint32
	PackedSize uint32
	Method     byte
}

type packedCacheItem struct {
	hash string
	data []byte
}

type Reader struct {
	file             *os.File
	Manifest         Manifest
	manifestOffset   int64
	chunks           map[string]chunkRecord
	packedCache      map[string]*list.Element
	packedCacheLRU   *list.List
	packedCacheBytes int64
}

func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*Reader, error) {
		_ = f.Close()
		return nil, err
	}

	info, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	if info.Size() < int64(len(fileMagic)+footerSize) {
		return fail(ErrInvalidArchive)
	}

	header := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(f, header); err != nil {
		return fail(err)
	}
	if !bytes.Equal(header, fileMagic[:]) {
		return fail(ErrInvalidArchive)
	}

	footer := make([]byte, footerSize)
	if _, err := f.ReadAt(footer, info.Size()-footerSize); err != nil {
		return fail(err)
	}
	if !bytes.Equal(footer[:4], footerMagic[:]) {
		return fail(ErrInvalidArchive)
	}
	manifestOffset := int64(binary.LittleEndian.Uint64(footer[4:12]))
	if manifestOffset < int64(len(fileMagic)) || manifestOffset >= info.Size()-footerSize {
		return fail(ErrInvalidArchive)
	}

	manifest, err := readManifestAt(f, manifestOffset, footer[12:44])
	if err != nil {
		return fail(err)
	}
	chunks, err := scanChunkIndex(f, manifestOffset)
	if err != nil {
		return fail(err)
	}

	return &Reader{
		file:           f,
		Manifest:       manifest,
		manifestOffset: manifestOffset,
		chunks:         chunks,
		packedCache:    make(map[string]*list.Element),
		packedCacheLRU: list.New(),
	}, nil
}

func (r *Reader) Close() error {
	return r.file.Close()
}

func readManifestAt(f *os.File, offset int64, expectedHash []byte) (Manifest, error) {
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return Manifest{}, err
	}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return Manifest{}, err
	}

	var data []byte
	switch {
	case bytes.Equal(magic, manifestMagicV1[:]):
		var size uint64
		if err := binary.Read(f, binary.LittleEndian, &size); err != nil {
			return Manifest{}, err
		}
		if size > maxManifestSize {
			return Manifest{}, fmt.Errorf("каталог архива слишком большой")
		}
		data = make([]byte, size)
		if _, err := io.ReadFull(f, data); err != nil {
			return Manifest{}, err
		}
	case bytes.Equal(magic, manifestMagicV2[:]):
		var packedSize uint64
		var rawSize uint64
		if err := binary.Read(f, binary.LittleEndian, &packedSize); err != nil {
			return Manifest{}, err
		}
		if err := binary.Read(f, binary.LittleEndian, &rawSize); err != nil {
			return Manifest{}, err
		}
		methodBuf := []byte{0}
		if _, err := io.ReadFull(f, methodBuf); err != nil {
			return Manifest{}, err
		}
		if packedSize > maxManifestSize || rawSize > maxManifestSize {
			return Manifest{}, fmt.Errorf("каталог архива слишком большой")
		}
		packed := make([]byte, packedSize)
		if _, err := io.ReadFull(f, packed); err != nil {
			return Manifest{}, err
		}
		var err error
		data, err = unpackData(methodBuf[0], packed, uint32(rawSize))
		if err != nil {
			return Manifest{}, fmt.Errorf("не удалось прочитать каталог архива: %w", err)
		}
	default:
		return Manifest{}, ErrInvalidArchive
	}

	digest := sha256.Sum256(data)
	if !bytes.Equal(digest[:], expectedHash) {
		return Manifest{}, fmt.Errorf("SHA-256 каталога архива не совпадает")
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, err
	}
	if manifest.Format != "deduparc" || (manifest.Version != 1 && manifest.Version != 2) {
		return Manifest{}, fmt.Errorf("неподдерживаемая версия формата")
	}
	return manifest, nil
}

func scanChunkIndex(f *os.File, manifestOffset int64) (map[string]chunkRecord, error) {
	chunks := make(map[string]chunkRecord)
	offset := int64(len(fileMagic))
	for offset < manifestOffset {
		header := make([]byte, 4+32+4+4+1)
		if _, err := f.ReadAt(header, offset); err != nil {
			return nil, err
		}
		if !bytes.Equal(header[:4], chunkMagic[:]) {
			return nil, fmt.Errorf("повреждённая запись данных по смещению %d", offset)
		}
		hash := hex.EncodeToString(header[4:36])
		rawSize := binary.LittleEndian.Uint32(header[36:40])
		packedSize := binary.LittleEndian.Uint32(header[40:44])
		method := header[44]
		dataOffset := offset + int64(len(header))
		next := dataOffset + int64(packedSize)
		if next > manifestOffset || next < dataOffset {
			return nil, ErrInvalidArchive
		}
		if _, exists := chunks[hash]; exists {
			return nil, fmt.Errorf("дубликат физического блока %s", hash)
		}
		chunks[hash] = chunkRecord{
			Offset:     dataOffset,
			RawSize:    rawSize,
			PackedSize: packedSize,
			Method:     method,
		}
		offset = next
	}
	if offset != manifestOffset {
		return nil, ErrInvalidArchive
	}
	return chunks, nil
}

func (r *Reader) readChunk(hash string) ([]byte, error) {
	record, ok := r.chunks[hash]
	if !ok {
		return nil, fmt.Errorf("в архиве нет блока %s", hash)
	}
	packed := make([]byte, record.PackedSize)
	if _, err := r.file.ReadAt(packed, record.Offset); err != nil {
		return nil, err
	}

	data, err := unpackData(record.Method, packed, record.RawSize)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != hash {
		return nil, fmt.Errorf("SHA-256 блока %s не совпадает", hash)
	}
	return data, nil
}

func unpackData(method byte, packed []byte, rawSize uint32) ([]byte, error) {
	var data []byte
	switch method {
	case methodRaw:
		data = append([]byte(nil), packed...)
	case methodZlib:
		zr, err := zlib.NewReader(bytes.NewReader(packed))
		if err != nil {
			return nil, err
		}
		data, err = io.ReadAll(io.LimitReader(zr, int64(rawSize)+1))
		closeErr := zr.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
	default:
		return nil, fmt.Errorf("неизвестный метод сжатия %d", method)
	}
	if len(data) != int(rawSize) {
		return nil, fmt.Errorf("размер распакованных данных не совпадает")
	}
	return data, nil
}

const packedCacheLimit = int64(64 * 1024 * 1024)

func (r *Reader) readPackedChunk(hash string) ([]byte, error) {
	if elem, ok := r.packedCache[hash]; ok {
		r.packedCacheLRU.MoveToBack(elem)
		return elem.Value.(packedCacheItem).data, nil
	}
	data, err := r.readChunk(hash)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > packedCacheLimit {
		return data, nil
	}
	for r.packedCacheBytes+int64(len(data)) > packedCacheLimit && r.packedCacheLRU.Len() > 0 {
		front := r.packedCacheLRU.Front()
		item := front.Value.(packedCacheItem)
		delete(r.packedCache, item.hash)
		r.packedCacheBytes -= int64(len(item.data))
		r.packedCacheLRU.Remove(front)
	}
	item := packedCacheItem{hash: hash, data: data}
	elem := r.packedCacheLRU.PushBack(item)
	r.packedCache[hash] = elem
	r.packedCacheBytes += int64(len(data))
	return data, nil
}

func (r *Reader) entryData(entry Entry, emit func([]byte) error) error {
	if entry.Packed != nil {
		data, err := r.readPackedChunk(entry.Packed.Chunk)
		if err != nil {
			return err
		}
		start := uint64(entry.Packed.Offset)
		end := start + uint64(entry.Packed.Length)
		if end < start || end > uint64(len(data)) {
			return fmt.Errorf("ссылка на упакованный файл выходит за границы блока")
		}
		return emit(data[start:end])
	}
	for _, hash := range entry.Chunks {
		data, err := r.readChunk(hash)
		if err != nil {
			return err
		}
		if err := emit(data); err != nil {
			return err
		}
	}
	return nil
}

// StoredSize возвращает приблизительный физический объём данных файла внутри архива.
// Для файлов из общего блока значение распределяется пропорционально их длине.
func (r *Reader) StoredSize(entry Entry) int64 {
	if entry.Packed != nil {
		record, ok := r.chunks[entry.Packed.Chunk]
		if !ok || record.RawSize == 0 || entry.Packed.Length == 0 {
			return 0
		}
		size := int64(record.PackedSize) * int64(entry.Packed.Length) / int64(record.RawSize)
		if size == 0 && record.PackedSize > 0 {
			return 1
		}
		return size
	}
	var size int64
	seen := make(map[string]struct{}, len(entry.Chunks))
	for _, hash := range entry.Chunks {
		if _, ok := seen[hash]; ok {
			continue
		}
		seen[hash] = struct{}{}
		if record, ok := r.chunks[hash]; ok {
			size += int64(record.PackedSize)
		}
	}
	return size
}
