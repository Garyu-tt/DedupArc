package chunker

import (
	"fmt"
	"io"
	"math/bits"
)

const (
	tableSeed  uint64 = 0x9e3779b97f4a7c15
	windowSize        = 32
)

var hashTable = makeHashTable()

type Config struct {
	Min int `json:"min"`
	Avg int `json:"avg"`
	Max int `json:"max"`
}

func DefaultConfig() Config {
	return Config{
		Min: 64 * 1024,
		Avg: 256 * 1024,
		Max: 1024 * 1024,
	}
}

func (c Config) Validate() error {
	if c.Min <= 0 || c.Avg <= 0 || c.Max <= 0 {
		return fmt.Errorf("размеры чанков должны быть больше нуля")
	}
	if c.Min > c.Avg || c.Avg > c.Max {
		return fmt.Errorf("ожидается Min <= Avg <= Max")
	}
	if c.Avg&(c.Avg-1) != 0 {
		return fmt.Errorf("Avg должен быть степенью двойки")
	}
	return nil
}

// Split режет поток по скользящему хэшу. После небольшой вставки или удаления
// окно снова синхронизируется с прежним содержимым и возвращает старые границы.
func Split(r io.Reader, cfg Config, emit func([]byte) error) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	buf := make([]byte, 64*1024)
	chunk := make([]byte, 0, cfg.Max)
	mask := uint64(cfg.Avg - 1)
	var window [windowSize]byte
	windowPos := 0
	windowFilled := 0
	var hash uint64

	flush := func() error {
		if len(chunk) == 0 {
			return nil
		}
		out := append([]byte(nil), chunk...)
		chunk = chunk[:0]
		return emit(out)
	}

	for {
		n, err := r.Read(buf)
		if n > 0 {
			for _, b := range buf[:n] {
				chunk = append(chunk, b)

				if windowFilled < windowSize {
					hash = bits.RotateLeft64(hash, 1) ^ hashTable[b]
					window[windowPos] = b
					windowPos = (windowPos + 1) % windowSize
					windowFilled++
				} else {
					old := window[windowPos]
					window[windowPos] = b
					windowPos = (windowPos + 1) % windowSize
					hash = bits.RotateLeft64(hash, 1) ^ bits.RotateLeft64(hashTable[old], windowSize) ^ hashTable[b]
				}

				if len(chunk) >= cfg.Max || (len(chunk) >= cfg.Min && windowFilled == windowSize && hash&mask == 0) {
					if err := flush(); err != nil {
						return err
					}
				}
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	return flush()
}

func makeHashTable() [256]uint64 {
	var table [256]uint64
	x := tableSeed
	for i := range table {
		x += 0x9e3779b97f4a7c15
		z := x
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		table[i] = z ^ (z >> 31)
	}
	return table
}
