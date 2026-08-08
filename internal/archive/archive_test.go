package archive

import (
	"bytes"
	"compress/zlib"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Garyu-tt/deduparc/internal/chunker"
)

func TestCreateOpenExtractVerifyAndDeduplicate(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}

	common := bytes.Repeat([]byte("одинаковый блок данных\n"), 30000)
	if err := os.WriteFile(filepath.Join(source, "a.txt"), common, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "b.txt"), common, 0o644); err != nil {
		t.Fatal(err)
	}
	modified := append([]byte("небольшое изменение\n"), common...)
	if err := os.WriteFile(filepath.Join(source, "c.txt"), modified, 0o644); err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(root, "test.darc")
	result, err := Create(archivePath, []string{source}, CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.Stats.UniqueChunks >= result.Manifest.Stats.Chunks {
		t.Fatalf("дедупликация не сработала: unique=%d total=%d",
			result.Manifest.Stats.UniqueChunks, result.Manifest.Stats.Chunks)
	}

	r, err := Open(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Verify(); err != nil {
		t.Fatal(err)
	}

	target := filepath.Join(root, "out")
	if err := r.Extract(target); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(target, "source", "c.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, modified) {
		t.Fatal("распакованный файл отличается от исходного")
	}
}

func TestCreateArchiveInsideSourceDirectory(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "data.txt"), bytes.Repeat([]byte("x"), 200000), 0o644); err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(root, "inside.darc")
	if _, err := Create(archivePath, []string{root}, CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	r, err := Open(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	for _, entry := range r.Manifest.Entries {
		if filepath.Base(entry.Path) == "inside.darc" || strings.Contains(entry.Path, ".inside.darc.tmp-") {
			t.Fatalf("архив включил собственный выходной файл: %s", entry.Path)
		}
	}
}

func TestSafeJoinRejectsTraversal(t *testing.T) {
	base := t.TempDir()
	for _, path := range []string{"../evil", "../../evil", "..\\evil"} {
		if _, err := safeJoin(base, path); err == nil && filepath.Separator == '\\' {
			t.Fatalf("путь %q должен быть отклонён", path)
		}
	}
}

func TestReadCatalogDoesNotScanDataBlocks(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "data.bin"), bytes.Repeat([]byte("abc123"), 100000), 0o644); err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(root, "catalog.darc")
	if _, err := Create(archivePath, []string{source}, CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	f, err := os.OpenFile(archivePath, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Повреждаем сигнатуру первой записи данных. Каталог в хвосте остаётся целым.
	if _, err := f.WriteAt([]byte("BAD!"), int64(len(fileMagic))); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	catalog, err := ReadCatalog(archivePath)
	if err != nil {
		t.Fatalf("каталог должен читаться без полного сканирования данных: %v", err)
	}
	if catalog.Manifest.Stats.Files != 1 {
		t.Fatalf("ожидался один файл, получено %d", catalog.Manifest.Stats.Files)
	}
	if _, err := Open(archivePath); err == nil {
		t.Fatal("полное открытие должно обнаружить повреждённую запись данных")
	}
}

func TestCreateEmptyArchive(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "empty.darc")

	if _, err := Create(archivePath, nil, CreateOptions{}); err != nil {
		t.Fatalf("Create(): %v", err)
	}

	r, err := Open(archivePath)
	if err != nil {
		t.Fatalf("Open(): %v", err)
	}
	defer r.Close()

	if len(r.Manifest.Entries) != 0 {
		t.Fatalf("ожидался пустой архив, получено записей: %d", len(r.Manifest.Entries))
	}
	if _, err := r.Verify(); err != nil {
		t.Fatalf("Verify(): %v", err)
	}
}

func TestCreateReportsProgressAndCanBeCancelled(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source.bin")
	if err := os.WriteFile(source, bytes.Repeat([]byte("progress-data-"), 50000), 0o644); err != nil {
		t.Fatal(err)
	}

	var phases []CreatePhase
	archivePath := filepath.Join(root, "progress.darc")
	_, err := Create(archivePath, []string{source}, CreateOptions{Progress: func(p CreateProgress) bool {
		if len(phases) == 0 || phases[len(phases)-1] != p.Phase {
			phases = append(phases, p.Phase)
		}
		return p.Phase != PhaseWriting
	}})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("ожидалась отмена, получено: %v", err)
	}
	if len(phases) < 3 || phases[0] != PhasePreparing || phases[1] != PhaseCompressing || phases[2] != PhaseWriting {
		t.Fatalf("неожиданная последовательность этапов: %v", phases)
	}
	if _, err := os.Stat(archivePath); !os.IsNotExist(err) {
		t.Fatalf("отменённая операция не должна оставлять готовый архив: %v", err)
	}
}

func TestSmallFilesArePackedAndRoundTrip(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "small")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}

	const count = 2500
	for i := 0; i < count; i++ {
		body := []byte(fmt.Sprintf("{\"id\":%d,\"kind\":\"item\",\"enabled\":true,\"padding\":\"%s\"}\n", i%400, strings.Repeat("abc123", 24)))
		if err := os.WriteFile(filepath.Join(source, fmt.Sprintf("item-%04d.json", i)), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	archivePath := filepath.Join(root, "small.darc")
	result, err := Create(archivePath, []string{source}, CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.Version != 2 {
		t.Fatalf("ожидалась версия формата 2, получена %d", result.Manifest.Version)
	}
	if result.Manifest.Stats.SmallFiles != count {
		t.Fatalf("ожидалось %d мелких файлов, получено %d", count, result.Manifest.Stats.SmallFiles)
	}
	if result.Manifest.Stats.SmallPacks >= count/10 {
		t.Fatalf("мелкие файлы почти не объединились: packs=%d files=%d", result.Manifest.Stats.SmallPacks, count)
	}
	if result.Manifest.Stats.ManifestStoredBytes >= result.Manifest.Stats.ManifestRawBytes {
		t.Fatalf("каталог архива не сжался: raw=%d stored=%d", result.Manifest.Stats.ManifestRawBytes, result.Manifest.Stats.ManifestStoredBytes)
	}

	r, err := Open(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Verify(); err != nil {
		t.Fatal(err)
	}

	out := filepath.Join(root, "out")
	if err := r.Extract(out); err != nil {
		t.Fatal(err)
	}
	for _, idx := range []int{0, 137, 2499} {
		want, err := os.ReadFile(filepath.Join(source, fmt.Sprintf("item-%04d.json", idx)))
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(filepath.Join(out, "small", fmt.Sprintf("item-%04d.json", idx)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("файл %d после распаковки отличается", idx)
		}
	}
}

func TestMaximumCompressionIsNotLargerThanFastOnCompressibleData(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "data.bin")
	data := bytes.Repeat([]byte("the-quick-brown-fox-jumps-over-the-lazy-dog-0123456789\n"), 120000)
	if err := os.WriteFile(source, data, 0o644); err != nil {
		t.Fatal(err)
	}

	fastPath := filepath.Join(root, "fast.darc")
	fast, err := Create(fastPath, []string{source}, CreateOptions{Compression: CompressionFast})
	if err != nil {
		t.Fatal(err)
	}
	maxPath := filepath.Join(root, "max.darc")
	maximum, err := Create(maxPath, []string{source}, CreateOptions{Compression: CompressionMaximum})
	if err != nil {
		t.Fatal(err)
	}
	if maximum.Bytes > fast.Bytes {
		t.Fatalf("maximum получился больше fast: maximum=%d fast=%d", maximum.Bytes, fast.Bytes)
	}
}

func TestOpenLegacyV1Archive(t *testing.T) {
	root := t.TempDir()
	archivePath := filepath.Join(root, "legacy.darc")
	data := []byte("legacy deduparc v1 data\n")
	digest := sha256.Sum256(data)

	f, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(fileMagic[:]); err != nil {
		t.Fatal(err)
	}
	method, packed, err := compressBlock(data, zlib.BestSpeed, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeChunkRecord(f, digest, uint32(len(data)), method, packed); err != nil {
		t.Fatal(err)
	}
	manifestOffset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatal(err)
	}
	manifest := Manifest{
		Format:    "deduparc",
		Version:   1,
		CreatedAt: "2026-08-08T00:00:00Z",
		Chunker:   chunkerInfo(chunker.DefaultConfig()),
		Entries: []Entry{{
			Path:         "legacy.txt",
			Type:         "file",
			Size:         int64(len(data)),
			Mode:         0o644,
			ModifiedUnix: 0,
			SHA256:       hex.EncodeToString(digest[:]),
			Chunks:       []string{hex.EncodeToString(digest[:])},
		}},
		Stats: Stats{Files: 1, Chunks: 1, UniqueChunks: 1, SourceBytes: int64(len(data)), UniqueRawBytes: int64(len(data)), StoredBytes: int64(len(packed))},
	}
	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	manifestHash := sha256.Sum256(manifestBytes)
	if _, err := f.Write(manifestMagicV1[:]); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(manifestBytes))); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(manifestBytes); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(footerMagic[:]); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(manifestOffset)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(manifestHash[:]); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := Open(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Verify(); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(root, "legacy-out")
	if err := r.Extract(out); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(out, "legacy.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("v1 архив распаковался неверно")
	}
}

func TestSmallFilePipelineSingleAndParallelWorkers(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "many")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1800; i++ {
		body := []byte(fmt.Sprintf("{\"id\":%d,\"group\":%d,\"value\":\"%s\"}\n", i, i%31, strings.Repeat("xyz123", 18)))
		if err := os.WriteFile(filepath.Join(source, fmt.Sprintf("item-%05d.json", i)), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	onePath := filepath.Join(root, "one.darc")
	one, err := Create(onePath, []string{source}, CreateOptions{IOWorkers: 1})
	if err != nil {
		t.Fatal(err)
	}
	parallelPath := filepath.Join(root, "parallel.darc")
	parallel, err := Create(parallelPath, []string{source}, CreateOptions{IOWorkers: 32})
	if err != nil {
		t.Fatal(err)
	}
	if one.Manifest.Stats.SmallFiles != parallel.Manifest.Stats.SmallFiles || one.Manifest.Stats.SmallPacks != parallel.Manifest.Stats.SmallPacks {
		t.Fatalf("параллельное чтение изменило раскладку мелких файлов: one=%+v parallel=%+v", one.Manifest.Stats, parallel.Manifest.Stats)
	}
	if one.Manifest.Stats.StoredBytes != parallel.Manifest.Stats.StoredBytes {
		t.Fatalf("параллельное чтение изменило размер данных: one=%d parallel=%d", one.Manifest.Stats.StoredBytes, parallel.Manifest.Stats.StoredBytes)
	}

	r, err := Open(parallelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Verify(); err != nil {
		t.Fatal(err)
	}
}

func TestPreparingScanCanBeCancelled(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "scan")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 700; i++ {
		if err := os.WriteFile(filepath.Join(source, fmt.Sprintf("f-%04d.txt", i)), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	archivePath := filepath.Join(root, "cancelled.darc")
	_, err := Create(archivePath, []string{source}, CreateOptions{Progress: func(p CreateProgress) bool {
		return !(p.Phase == PhasePreparing && p.FilesDone >= 512)
	}})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("ожидалась отмена во время сканирования, получено: %v", err)
	}
	if _, statErr := os.Stat(archivePath); !os.IsNotExist(statErr) {
		t.Fatalf("после отмены сканирования архив не должен появиться: %v", statErr)
	}
}
