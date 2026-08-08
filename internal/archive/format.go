package archive

import "errors"

var (
	fileMagic       = [8]byte{'D', 'A', 'R', 'C', 0, 1, '\r', '\n'}
	chunkMagic      = [4]byte{'C', 'H', 'N', 'K'}
	manifestMagicV1 = [4]byte{'M', 'N', 'F', 'T'}
	manifestMagicV2 = [4]byte{'M', 'N', 'F', '2'}
	footerMagic     = [4]byte{'D', 'E', 'N', 'D'}
)

const (
	methodRaw  byte = 0
	methodZlib byte = 1
	footerSize      = 4 + 8 + 32

	defaultSmallFileThreshold = 256 * 1024
	defaultSmallPackTarget    = 16 * 1024 * 1024
	maxManifestSize           = 512 * 1024 * 1024
)

var ErrInvalidArchive = errors.New("неверный или повреждённый архив")
