package archive

import (
	"bytes"
	"compress/zlib"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Garyu-tt/deduparc/internal/chunker"
)

type pendingSmallObject struct {
	digest       [32]byte
	offset       uint32
	length       uint32
	entryIndexes []int
}

type smallPack struct {
	class   string
	data    []byte
	objects []*pendingSmallObject
}

type scanMeta struct {
	IsDir        bool
	Size         int64
	Mode         os.FileMode
	ModifiedUnix int64
	Symlink      bool
}

type sourceEntry struct {
	sourcePath   string
	storedPath   string
	isDir        bool
	size         int64
	mode         os.FileMode
	modifiedUnix int64
}

type smallReadResult struct {
	data []byte
	err  error
}

type writer struct {
	out            *os.File
	seenChunks     map[[32]byte]struct{}
	manifest       Manifest
	excludedPaths  map[string]struct{}
	progress       func(CreateProgress) bool
	lastProgress   time.Time
	lastPhase      CreatePhase
	processedBytes int64
	totalBytes     int64
	filesDone      int
	smallFilesDone int
	ioWorkers      int

	compression        CompressionMode
	chunkLevel         int
	packLevel          int
	smallFileThreshold int
	smallPackTarget    int
	smallPacks         map[string]*smallPack
	resolvedSmall      map[[32]byte]PackedRef
	pendingSmall       map[[32]byte]*pendingSmallObject
	deltaMatcher       *deltaMatcher
}

func Create(output string, inputs []string, options CreateOptions) (CreateResult, error) {
	options = normalizeCreateOptions(options)
	if err := options.Chunker.Validate(); err != nil {
		return CreateResult{}, err
	}
	const maxUint32 = uint64(^uint32(0))
	if uint64(options.Chunker.Max) > maxUint32 || uint64(options.SmallFileThreshold) > maxUint32 || uint64(options.SmallPackTarget) > maxUint32 {
		return CreateResult{}, fmt.Errorf("размер блока не должен превышать 4 ГиБ")
	}

	outputAbs, err := filepath.Abs(output)
	if err != nil {
		return CreateResult{}, err
	}
	if !notifyProgress(options.Progress, CreateProgress{Phase: PhasePreparing}) {
		return CreateResult{}, ErrCancelled
	}
	scanned, totalBytes, err := scanInputs(inputs, map[string]struct{}{outputAbs: {}}, func(path string, count int) bool {
		return notifyProgress(options.Progress, CreateProgress{Phase: PhasePreparing, CurrentPath: path, FilesDone: count})
	})
	if err != nil {
		return CreateResult{}, err
	}
	if !notifyProgress(options.Progress, CreateProgress{Phase: PhasePreparing, TotalBytes: totalBytes, FilesDone: len(scanned)}) {
		return CreateResult{}, ErrCancelled
	}

	if err := os.MkdirAll(filepath.Dir(outputAbs), 0o755); err != nil {
		return CreateResult{}, err
	}
	out, err := os.CreateTemp(filepath.Dir(outputAbs), "."+filepath.Base(outputAbs)+".tmp-*")
	if err != nil {
		return CreateResult{}, err
	}
	tempAbs, err := filepath.Abs(out.Name())
	if err != nil {
		_ = out.Close()
		_ = os.Remove(out.Name())
		return CreateResult{}, err
	}
	committed := false
	defer func() {
		_ = out.Close()
		if !committed {
			_ = os.Remove(tempAbs)
		}
	}()

	chunkLevel, packLevel := compressionLevels(options.Compression)
	w := &writer{
		out:                out,
		seenChunks:         make(map[[32]byte]struct{}),
		excludedPaths:      map[string]struct{}{outputAbs: {}, tempAbs: {}},
		progress:           options.Progress,
		totalBytes:         totalBytes,
		ioWorkers:          options.IOWorkers,
		compression:        options.Compression,
		chunkLevel:         chunkLevel,
		packLevel:          packLevel,
		smallFileThreshold: options.SmallFileThreshold,
		smallPackTarget:    options.SmallPackTarget,
		smallPacks:         make(map[string]*smallPack),
		resolvedSmall:      make(map[[32]byte]PackedRef),
		pendingSmall:       make(map[[32]byte]*pendingSmallObject),
		manifest: Manifest{
			Format:    "deduparc",
			Version:   archiveVersion(options.Compression),
			CreatedAt: time.Now().UTC().Format(time.RFC3339),
			Chunker:   chunkerInfo(options.Chunker),
			Compression: CompressionInfo{
				Mode:               options.Compression,
				SmallFileThreshold: options.SmallFileThreshold,
				SmallPackTarget:    options.SmallPackTarget,
			},
		},
	}
	if options.Compression == CompressionExtreme {
		w.deltaMatcher = newDeltaMatcher()
	}
	w.manifest.Entries = make([]Entry, 0, len(scanned))

	if _, err := out.Write(fileMagic[:]); err != nil {
		return CreateResult{}, err
	}

	if err := w.addScanned(scanned, options.Chunker); err != nil {
		return CreateResult{}, err
	}
	if err := w.flushAllSmallPacks(); err != nil {
		return CreateResult{}, err
	}

	sort.Slice(w.manifest.Entries, func(i, j int) bool {
		return w.manifest.Entries[i].Path < w.manifest.Entries[j].Path
	})

	if !w.notify(PhaseWriting, "") {
		return CreateResult{}, ErrCancelled
	}
	manifestOffset, err := out.Seek(0, io.SeekCurrent)
	if err != nil {
		return CreateResult{}, err
	}
	manifestRaw, manifestMethod, manifestPacked, err := encodeManifestV2(&w.manifest)
	if err != nil {
		return CreateResult{}, err
	}
	manifestHash := sha256.Sum256(manifestRaw)

	if _, err := out.Write(manifestMagicV2[:]); err != nil {
		return CreateResult{}, err
	}
	if err := binary.Write(out, binary.LittleEndian, uint64(len(manifestPacked))); err != nil {
		return CreateResult{}, err
	}
	if err := binary.Write(out, binary.LittleEndian, uint64(len(manifestRaw))); err != nil {
		return CreateResult{}, err
	}
	if _, err := out.Write([]byte{manifestMethod}); err != nil {
		return CreateResult{}, err
	}
	if _, err := out.Write(manifestPacked); err != nil {
		return CreateResult{}, err
	}
	if _, err := out.Write(footerMagic[:]); err != nil {
		return CreateResult{}, err
	}
	if err := binary.Write(out, binary.LittleEndian, uint64(manifestOffset)); err != nil {
		return CreateResult{}, err
	}
	if _, err := out.Write(manifestHash[:]); err != nil {
		return CreateResult{}, err
	}
	if err := out.Sync(); err != nil {
		return CreateResult{}, err
	}
	info, err := out.Stat()
	if err != nil {
		return CreateResult{}, err
	}
	if err := out.Close(); err != nil {
		return CreateResult{}, err
	}
	if err := replaceFile(tempAbs, outputAbs); err != nil {
		return CreateResult{}, err
	}
	committed = true
	if !w.notify(PhaseWriting, "") {
		return CreateResult{}, ErrCancelled
	}
	return CreateResult{Manifest: w.manifest, Bytes: info.Size()}, nil
}

func normalizeCreateOptions(options CreateOptions) CreateOptions {
	if options.Chunker == (chunker.Config{}) {
		options.Chunker = chunker.DefaultConfig()
	}
	if options.Compression == "" {
		options.Compression = CompressionBalanced
	}
	switch options.Compression {
	case CompressionFast, CompressionBalanced, CompressionMaximum, CompressionExtreme:
	default:
		options.Compression = CompressionBalanced
	}
	if options.SmallFileThreshold <= 0 {
		switch options.Compression {
		case CompressionFast:
			options.SmallFileThreshold = 128 * 1024
		default:
			options.SmallFileThreshold = defaultSmallFileThreshold
		}
	}
	if options.SmallPackTarget <= 0 {
		switch options.Compression {
		case CompressionFast:
			options.SmallPackTarget = 4 * 1024 * 1024
		case CompressionMaximum, CompressionExtreme:
			options.SmallPackTarget = 16 * 1024 * 1024
		default:
			options.SmallPackTarget = defaultSmallPackTarget
		}
	}
	if options.IOWorkers <= 0 {
		options.IOWorkers = runtime.GOMAXPROCS(0) * 4
		if options.IOWorkers < 16 {
			options.IOWorkers = 16
		}
		if options.IOWorkers > 64 {
			options.IOWorkers = 64
		}
	}
	if options.SmallPackTarget < options.SmallFileThreshold {
		options.SmallPackTarget = options.SmallFileThreshold
	}
	return options
}

func archiveVersion(mode CompressionMode) int {
	if mode == CompressionExtreme {
		return 3
	}
	return 2
}

func compressionLevels(mode CompressionMode) (chunkLevel, packLevel int) {
	switch mode {
	case CompressionFast:
		return zlib.BestSpeed, 3
	case CompressionMaximum, CompressionExtreme:
		return zlib.BestCompression, zlib.BestCompression
	default:
		return 8, zlib.BestCompression
	}
}

func (w *writer) addScanned(entries []sourceEntry, cfg chunker.Config) error {
	batch := make([]sourceEntry, 0, 512)
	var batchBytes int64

	flushBatch := func() error {
		if len(batch) == 0 {
			return nil
		}
		if err := w.addSmallBatch(batch); err != nil {
			return err
		}
		batch = batch[:0]
		batchBytes = 0
		return nil
	}

	for _, item := range entries {
		if item.isDir {
			w.manifest.Entries = append(w.manifest.Entries, Entry{
				Path:         item.storedPath,
				Type:         "dir",
				Mode:         uint32(item.mode.Perm()),
				ModifiedUnix: item.modifiedUnix,
			})
			w.manifest.Stats.Directories++
			continue
		}

		if item.size <= int64(w.smallFileThreshold) {
			batch = append(batch, item)
			batchBytes += item.size
			// Ограничиваем read-ahead по памяти, но держим достаточно запросов,
			// чтобы NVMe/SSD не простаивал на тысячах мелких файлов.
			if len(batch) >= 1024 || batchBytes >= 64*1024*1024 {
				if err := flushBatch(); err != nil {
					return err
				}
			}
			continue
		}

		if err := flushBatch(); err != nil {
			return err
		}
		if err := w.addLargeFile(item, cfg); err != nil {
			return err
		}
	}
	return flushBatch()
}

func (w *writer) addSmallBatch(batch []sourceEntry) error {
	results := make([]smallReadResult, len(batch))
	jobs := make(chan int)
	workers := w.ioWorkers
	if workers > len(batch) {
		workers = len(batch)
	}
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			defer wg.Done()
			for i := range jobs {
				data, err := os.ReadFile(batch[i].sourcePath)
				if err == nil && int64(len(data)) != batch[i].size {
					err = fmt.Errorf("%s: размер файла изменился во время архивации", batch[i].storedPath)
				}
				results[i] = smallReadResult{data: data, err: err}
			}
		}()
	}
	for i := range batch {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	// Результаты добавляются в исходном порядке. Поэтому одинаковый набор данных
	// даёт стабильную раскладку pack-блоков независимо от порядка завершения I/O.
	for i, item := range batch {
		if results[i].err != nil {
			return results[i].err
		}
		if err := w.addSmallData(item, results[i].data); err != nil {
			return err
		}
	}
	return nil
}

func (w *writer) addLargeFile(item sourceEntry, cfg chunker.Config) error {
	f, err := os.Open(item.sourcePath)
	if err != nil {
		return err
	}
	defer f.Close()

	entry := Entry{
		Path:         item.storedPath,
		Type:         "file",
		Size:         item.size,
		Mode:         uint32(item.mode.Perm()),
		ModifiedUnix: item.modifiedUnix,
	}
	fileHash := sha256.New()
	entropySkip := isProbablyPrecompressedPath(item.storedPath)

	err = chunker.Split(f, cfg, func(data []byte) error {
		_, _ = fileHash.Write(data)
		digest := sha256.Sum256(data)
		entry.Chunks = append(entry.Chunks, hex.EncodeToString(digest[:]))
		w.manifest.Stats.Chunks++

		if err := w.storeChunk(digest, data, w.chunkLevel, entropySkip); err != nil {
			return err
		}

		w.processedBytes += int64(len(data))
		if !w.notify(PhaseCompressing, item.storedPath) {
			return ErrCancelled
		}
		return nil
	})
	if err != nil {
		return err
	}

	entry.SHA256 = hex.EncodeToString(fileHash.Sum(nil))
	w.manifest.Entries = append(w.manifest.Entries, entry)
	w.manifest.Stats.Files++
	w.manifest.Stats.SourceBytes += item.size
	w.filesDone++
	if !w.notify(PhaseCompressing, item.storedPath) {
		return ErrCancelled
	}
	return nil
}

func (w *writer) addSmallData(item sourceEntry, data []byte) error {
	digest := sha256.Sum256(data)
	entry := Entry{
		Path:         item.storedPath,
		Type:         "file",
		Size:         item.size,
		Mode:         uint32(item.mode.Perm()),
		ModifiedUnix: item.modifiedUnix,
	}
	w.manifest.Entries = append(w.manifest.Entries, entry)
	entryIndex := len(w.manifest.Entries) - 1
	w.manifest.Stats.Files++
	w.manifest.Stats.SmallFiles++
	w.manifest.Stats.SourceBytes += item.size
	w.filesDone++
	w.smallFilesDone++
	w.processedBytes += item.size

	if len(data) > 0 {
		w.manifest.Stats.Chunks++
		if ref, ok := w.resolvedSmall[digest]; ok {
			copyRef := ref
			w.manifest.Entries[entryIndex].Packed = &copyRef
		} else if pending, ok := w.pendingSmall[digest]; ok {
			pending.entryIndexes = append(pending.entryIndexes, entryIndex)
		} else {
			class := smallFileClass(item.storedPath, data)
			pack := w.smallPacks[class]
			if pack == nil {
				pack = &smallPack{class: class, data: make([]byte, 0, w.smallPackTarget)}
				w.smallPacks[class] = pack
			}
			if len(pack.data) > 0 && len(pack.data)+len(data) > w.smallPackTarget {
				if err := w.flushSmallPack(pack); err != nil {
					return err
				}
			}
			obj := &pendingSmallObject{
				digest:       digest,
				offset:       uint32(len(pack.data)),
				length:       uint32(len(data)),
				entryIndexes: []int{entryIndex},
			}
			pack.data = append(pack.data, data...)
			pack.objects = append(pack.objects, obj)
			w.pendingSmall[digest] = obj
			if len(pack.data) >= w.smallPackTarget {
				if err := w.flushSmallPack(pack); err != nil {
					return err
				}
			}
		}
	}

	if !w.notify(PhaseCompressing, item.storedPath) {
		return ErrCancelled
	}
	return nil
}

func (w *writer) flushAllSmallPacks() error {
	classes := make([]string, 0, len(w.smallPacks))
	for class := range w.smallPacks {
		classes = append(classes, class)
	}
	sort.Strings(classes)
	for _, class := range classes {
		if err := w.flushSmallPack(w.smallPacks[class]); err != nil {
			return err
		}
	}
	return nil
}

func (w *writer) flushSmallPack(pack *smallPack) error {
	if pack == nil || len(pack.data) == 0 {
		return nil
	}
	digest := sha256.Sum256(pack.data)
	if err := w.storeChunk(digest, pack.data, w.packLevel, pack.class == "compressed"); err != nil {
		return err
	}
	chunkHash := hex.EncodeToString(digest[:])
	for _, obj := range pack.objects {
		ref := PackedRef{Chunk: chunkHash, Offset: obj.offset, Length: obj.length}
		w.resolvedSmall[obj.digest] = ref
		delete(w.pendingSmall, obj.digest)
		for _, entryIndex := range obj.entryIndexes {
			copyRef := ref
			w.manifest.Entries[entryIndex].Packed = &copyRef
		}
	}
	w.manifest.Stats.SmallPacks++
	pack.data = pack.data[:0]
	pack.objects = pack.objects[:0]
	return nil
}

func (w *writer) storeChunk(digest [32]byte, data []byte, level int, entropySkip bool) error {
	if _, exists := w.seenChunks[digest]; exists {
		return nil
	}
	w.seenChunks[digest] = struct{}{}
	w.manifest.Stats.UniqueChunks++
	w.manifest.Stats.UniqueRawBytes += int64(len(data))

	method, packed, err := compressBlock(data, level, entropySkip || w.compression == CompressionFast)
	if err != nil {
		return err
	}
	if w.deltaMatcher != nil {
		deltaMethod, deltaPacked, ok, err := w.deltaMatcher.try(data, len(packed))
		if err != nil {
			return err
		}
		if ok {
			method, packed = deltaMethod, deltaPacked
			w.manifest.Stats.DeltaChunks++
		} else {
			w.deltaMatcher.offer(digest, data)
		}
	}
	w.manifest.Stats.StoredBytes += int64(len(packed))
	return writeChunkRecord(w.out, digest, uint32(len(data)), method, packed)
}

func (w *writer) notify(phase CreatePhase, currentPath string) bool {
	if w.progress == nil {
		return true
	}
	now := time.Now()
	force := phase != w.lastPhase || w.processedBytes >= w.totalBytes
	if !force && !w.lastProgress.IsZero() && now.Sub(w.lastProgress) < 50*time.Millisecond {
		return true
	}
	w.lastProgress = now
	w.lastPhase = phase
	return notifyProgress(w.progress, CreateProgress{
		Phase:          phase,
		CurrentPath:    currentPath,
		ProcessedBytes: w.processedBytes,
		TotalBytes:     w.totalBytes,
		StoredBytes:    w.manifest.Stats.StoredBytes,
		FilesDone:      w.filesDone,
		SmallFilesDone: w.smallFilesDone,
	})
}

func notifyProgress(callback func(CreateProgress) bool, progress CreateProgress) bool {
	if callback == nil {
		return true
	}
	return callback(progress)
}

func scanInputs(inputs []string, excluded map[string]struct{}, progress func(string, int) bool) ([]sourceEntry, int64, error) {
	entries := make([]sourceEntry, 0, 1024)
	paths := make(map[string]struct{}, 1024)
	var totalBytes int64
	seen := 0

	for _, input := range inputs {
		clean := filepath.Clean(input)
		base := filepath.Dir(clean)
		if err := walkSource(clean, func(path string, meta scanMeta) (bool, error) {
			absPath, err := filepath.Abs(path)
			if err != nil {
				return false, err
			}
			if _, skip := excluded[absPath]; skip {
				return meta.IsDir, nil
			}
			if meta.Symlink {
				return false, fmt.Errorf("символические ссылки пока не поддерживаются: %s", path)
			}
			rel, err := filepath.Rel(base, path)
			if err != nil {
				return false, err
			}
			storedPath := filepath.ToSlash(rel)
			if storedPath == "." || storedPath == "" {
				storedPath = filepath.Base(clean)
			}
			storedPath = strings.TrimPrefix(storedPath, "./")
			if _, exists := paths[storedPath]; exists {
				return false, fmt.Errorf("путь %q встречается в архиве больше одного раза", storedPath)
			}
			paths[storedPath] = struct{}{}

			if !meta.IsDir && !meta.Mode.IsRegular() {
				return false, fmt.Errorf("поддерживаются только обычные файлы и каталоги: %s", path)
			}
			entries = append(entries, sourceEntry{
				sourcePath:   path,
				storedPath:   storedPath,
				isDir:        meta.IsDir,
				size:         meta.Size,
				mode:         meta.Mode,
				modifiedUnix: meta.ModifiedUnix,
			})
			if !meta.IsDir {
				totalBytes += meta.Size
			}
			seen++
			if progress != nil && seen%512 == 0 && !progress(storedPath, seen) {
				return false, ErrCancelled
			}
			return false, nil
		}); err != nil {
			return nil, 0, err
		}
	}

	sort.Slice(entries, func(i, j int) bool { return entries[i].storedPath < entries[j].storedPath })
	return entries, totalBytes, nil
}

func compressBlock(data []byte, level int, entropySkip bool) (byte, []byte, error) {
	if len(data) == 0 {
		return methodRaw, nil, nil
	}
	if entropySkip && len(data) >= 32*1024 && looksIncompressible(data) {
		return methodRaw, append([]byte(nil), data...), nil
	}
	packed, err := zlibCompress(data, level)
	if err != nil {
		return 0, nil, err
	}
	// Несколько выигранных байтов не стоят дополнительной работы при чтении.
	if len(packed)+8 >= len(data) {
		return methodRaw, append([]byte(nil), data...), nil
	}
	return methodZlib, packed, nil
}

func zlibCompress(data []byte, level int) ([]byte, error) {
	var buf bytes.Buffer
	zw, err := zlib.NewWriterLevel(&buf, level)
	if err != nil {
		return nil, err
	}
	if _, err := zw.Write(data); err != nil {
		return nil, err
	}
	if err := zw.Close(); err != nil {
		return nil, err
	}
	return append([]byte(nil), buf.Bytes()...), nil
}

func looksIncompressible(data []byte) bool {
	const sampleLimit = 64 * 1024
	if len(data) < 4096 {
		return false
	}
	sample := data
	if len(sample) > sampleLimit {
		step := len(data) / sampleLimit
		if step < 1 {
			step = 1
		}
		buf := make([]byte, 0, sampleLimit)
		for i := 0; i < len(data) && len(buf) < sampleLimit; i += step {
			buf = append(buf, data[i])
		}
		sample = buf
	}
	var counts [256]int
	for _, b := range sample {
		counts[b]++
	}
	inv := 1.0 / float64(len(sample))
	entropy := 0.0
	for _, count := range counts {
		if count == 0 {
			continue
		}
		p := float64(count) * inv
		entropy -= p * math.Log2(p)
	}
	return entropy >= 7.985
}

func isProbablyPrecompressedPath(path string) bool {
	_, ok := compressedSmallExtensions[strings.ToLower(filepath.Ext(path))]
	return ok
}

func smallFileClass(path string, data []byte) string {
	ext := strings.ToLower(filepath.Ext(path))
	if _, ok := textSmallExtensions[ext]; ok {
		return "text"
	}
	if _, ok := compressedSmallExtensions[ext]; ok {
		return "compressed"
	}
	if _, ok := executableSmallExtensions[ext]; ok {
		return "executable"
	}
	if looksLikeText(data) {
		return "text"
	}
	return "binary"
}

func looksLikeText(data []byte) bool {
	if len(data) == 0 {
		return true
	}
	limit := len(data)
	if limit > 8192 {
		limit = 8192
	}
	bad := 0
	for _, b := range data[:limit] {
		if b == 0 {
			return false
		}
		if b < 0x09 || (b > 0x0D && b < 0x20) {
			bad++
		}
	}
	return bad*100 <= limit
}

var textSmallExtensions = map[string]struct{}{
	".txt": {}, ".md": {}, ".json": {}, ".jsonc": {}, ".xml": {}, ".yaml": {}, ".yml": {},
	".toml": {}, ".ini": {}, ".cfg": {}, ".conf": {}, ".properties": {}, ".csv": {}, ".tsv": {},
	".log": {}, ".go": {}, ".rs": {}, ".c": {}, ".h": {}, ".cpp": {}, ".hpp": {}, ".cc": {},
	".cs": {}, ".java": {}, ".kt": {}, ".kts": {}, ".py": {}, ".js": {}, ".mjs": {}, ".cjs": {},
	".ts": {}, ".tsx": {}, ".jsx": {}, ".html": {}, ".htm": {}, ".css": {}, ".scss": {}, ".less": {},
	".sql": {}, ".sh": {}, ".ps1": {}, ".bat": {}, ".cmd": {}, ".lua": {}, ".rb": {}, ".php": {},
}

var compressedSmallExtensions = map[string]struct{}{
	".zip": {}, ".rar": {}, ".7z": {}, ".gz": {}, ".bz2": {}, ".xz": {}, ".zst": {}, ".cab": {},
	".jpg": {}, ".jpeg": {}, ".png": {}, ".gif": {}, ".webp": {}, ".avif": {}, ".heic": {},
	".mp3": {}, ".ogg": {}, ".opus": {}, ".flac": {}, ".aac": {}, ".mp4": {}, ".mkv": {}, ".webm": {},
	".pdf": {}, ".apk": {}, ".jar": {}, ".docx": {}, ".xlsx": {}, ".pptx": {},
}

var executableSmallExtensions = map[string]struct{}{
	".exe": {}, ".dll": {}, ".sys": {}, ".so": {}, ".dylib": {}, ".bin": {}, ".dat": {}, ".class": {},
}

func encodeManifestV2(manifest *Manifest) (raw []byte, method byte, packed []byte, err error) {
	for i := 0; i < 4; i++ {
		raw, err = json.Marshal(manifest)
		if err != nil {
			return nil, 0, nil, err
		}
		packed, err = zlibCompress(raw, zlib.BestCompression)
		if err != nil {
			return nil, 0, nil, err
		}
		method = methodZlib
		if len(packed)+8 >= len(raw) {
			method = methodRaw
			packed = append([]byte(nil), raw...)
		}
		rawSize := int64(len(raw))
		storedSize := int64(len(packed))
		if manifest.Stats.ManifestRawBytes == rawSize && manifest.Stats.ManifestStoredBytes == storedSize {
			return raw, method, packed, nil
		}
		manifest.Stats.ManifestRawBytes = rawSize
		manifest.Stats.ManifestStoredBytes = storedSize
	}
	raw, err = json.Marshal(manifest)
	if err != nil {
		return nil, 0, nil, err
	}
	packed, err = zlibCompress(raw, zlib.BestCompression)
	if err != nil {
		return nil, 0, nil, err
	}
	method = methodZlib
	if len(packed)+8 >= len(raw) {
		method = methodRaw
		packed = append([]byte(nil), raw...)
	}
	return raw, method, packed, nil
}

func writeChunkRecord(w io.Writer, digest [32]byte, rawSize uint32, method byte, packed []byte) error {
	if _, err := w.Write(chunkMagic[:]); err != nil {
		return err
	}
	if _, err := w.Write(digest[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, rawSize); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(packed))); err != nil {
		return err
	}
	if _, err := w.Write([]byte{method}); err != nil {
		return err
	}
	_, err := w.Write(packed)
	return err
}

func replaceFile(temp, target string) error {
	if err := os.Rename(temp, target); err == nil {
		return nil
	}
	backup := target + ".replace-backup"
	_ = os.Remove(backup)
	if _, err := os.Stat(target); err != nil {
		return os.Rename(temp, target)
	}
	if err := os.Rename(target, backup); err != nil {
		return err
	}
	if err := os.Rename(temp, target); err != nil {
		_ = os.Rename(backup, target)
		return err
	}
	_ = os.Remove(backup)
	return nil
}
