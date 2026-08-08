package archive

import "github.com/Garyu-tt/deduparc/internal/chunker"

type Manifest struct {
	Format      string          `json:"format"`
	Version     int             `json:"version"`
	CreatedAt   string          `json:"created_at"`
	Chunker     ChunkerInfo     `json:"chunker"`
	Compression CompressionInfo `json:"compression,omitempty"`
	Entries     []Entry         `json:"entries"`
	Stats       Stats           `json:"stats"`
}

type ChunkerInfo struct {
	Algorithm string `json:"algorithm"`
	Min       int    `json:"min"`
	Avg       int    `json:"avg"`
	Max       int    `json:"max"`
}

func chunkerInfo(cfg chunker.Config) ChunkerInfo {
	return ChunkerInfo{
		Algorithm: "buzhash-v1",
		Min:       cfg.Min,
		Avg:       cfg.Avg,
		Max:       cfg.Max,
	}
}

type CompressionMode string

const (
	CompressionFast     CompressionMode = "fast"
	CompressionBalanced CompressionMode = "balanced"
	CompressionMaximum  CompressionMode = "maximum"
)

type CompressionInfo struct {
	Mode               CompressionMode `json:"mode,omitempty"`
	SmallFileThreshold int             `json:"small_file_threshold,omitempty"`
	SmallPackTarget    int             `json:"small_pack_target,omitempty"`
}

type PackedRef struct {
	Chunk  string `json:"chunk"`
	Offset uint32 `json:"offset,omitempty"`
	Length uint32 `json:"length"`
}

type Entry struct {
	Path         string     `json:"path"`
	Type         string     `json:"type"`
	Size         int64      `json:"size,omitempty"`
	Mode         uint32     `json:"mode"`
	ModifiedUnix int64      `json:"modified_unix"`
	SHA256       string     `json:"sha256,omitempty"`
	Chunks       []string   `json:"chunks,omitempty"`
	Packed       *PackedRef `json:"packed,omitempty"`
}

type Stats struct {
	Files               int   `json:"files"`
	Directories         int   `json:"directories"`
	Chunks              int   `json:"chunks"`
	UniqueChunks        int   `json:"unique_chunks"`
	SmallFiles          int   `json:"small_files,omitempty"`
	SmallPacks          int   `json:"small_packs,omitempty"`
	SourceBytes         int64 `json:"source_bytes"`
	UniqueRawBytes      int64 `json:"unique_raw_bytes"`
	StoredBytes         int64 `json:"stored_bytes"`
	ManifestRawBytes    int64 `json:"manifest_raw_bytes,omitempty"`
	ManifestStoredBytes int64 `json:"manifest_stored_bytes,omitempty"`
}

type CreatePhase string

const (
	PhasePreparing   CreatePhase = "preparing"
	PhaseCompressing CreatePhase = "compressing"
	PhaseWriting     CreatePhase = "writing"
	PhaseVerifying   CreatePhase = "verifying"
)

type CreateProgress struct {
	Phase          CreatePhase
	CurrentPath    string
	ProcessedBytes int64
	TotalBytes     int64
	StoredBytes    int64
	FilesDone      int
	SmallFilesDone int
}

type CreateOptions struct {
	Chunker            chunker.Config
	Compression        CompressionMode
	SmallFileThreshold int
	SmallPackTarget    int
	IOWorkers          int
	Progress           func(CreateProgress) bool
}

type CreateResult struct {
	Manifest Manifest
	Bytes    int64
}
