package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Garyu-tt/deduparc/internal/archive"
	"github.com/Garyu-tt/deduparc/internal/version"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printHelp()
		return nil
	}

	switch args[0] {
	case "create", "c":
		return commandCreate(args[1:])
	case "list", "l":
		return commandList(args[1:])
	case "extract", "x":
		return commandExtract(args[1:])
	case "verify", "v":
		return commandVerify(args[1:])
	case "info", "i":
		return commandInfo(args[1:])
	case "bridge-create":
		return commandBridgeCreate(args[1:])
	case "version", "--version", "-v":
		fmt.Println("DedupArc", version.Current)
		return nil
	case "help", "--help", "-h":
		printHelp()
		return nil
	default:
		return fmt.Errorf("неизвестная команда %q", args[0])
	}
}

func commandCreate(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("использование: deduparc create [--fast|--balanced|--maximum|--extreme] <архив.darc> <файл|каталог> [...]")
	}
	mode := archive.CompressionBalanced
	if strings.HasPrefix(args[0], "--") {
		switch strings.ToLower(args[0]) {
		case "--fast":
			mode = archive.CompressionFast
		case "--balanced":
			mode = archive.CompressionBalanced
		case "--maximum", "--max":
			mode = archive.CompressionMaximum
		case "--extreme":
			mode = archive.CompressionExtreme
		default:
			return fmt.Errorf("неизвестный режим сжатия %q", args[0])
		}
		args = args[1:]
	}
	if len(args) < 2 {
		return fmt.Errorf("не указан архив или входные данные")
	}
	output := ensureExtension(args[0])
	result, err := archive.Create(output, args[1:], archive.CreateOptions{Compression: mode})
	if err != nil {
		return err
	}

	s := result.Manifest.Stats
	fmt.Printf("Создан: %s\n", output)
	fmt.Printf("Файлов: %d, каталогов: %d\n", s.Files, s.Directories)
	fmt.Printf("Мелких файлов объединено: %d в %d блоков\n", s.SmallFiles, s.SmallPacks)
	if mode == archive.CompressionExtreme {
		fmt.Printf("Дельта-блоков: %d\n", s.DeltaChunks)
	}
	fmt.Printf("Исходный объём: %s\n", humanBytes(s.SourceBytes))
	fmt.Printf("Размер архива: %s\n", humanBytes(result.Bytes))
	if s.SourceBytes > 0 {
		saved := 100 - float64(result.Bytes)*100/float64(s.SourceBytes)
		fmt.Printf("Экономия относительно исходных файлов: %.1f%%\n", saved)
	}
	return nil
}

func commandList(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("использование: deduparc list <архив.darc>")
	}
	r, err := archive.Open(args[0])
	if err != nil {
		return err
	}
	defer r.Close()

	entries := append([]archive.Entry(nil), r.Manifest.Entries...)
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	for _, e := range entries {
		kind := "DIR "
		size := ""
		if e.Type == "file" {
			kind = "FILE"
			size = humanBytes(e.Size)
		}
		fmt.Printf("%-4s %10s  %s\n", kind, size, e.Path)
	}
	return nil
}

func commandExtract(args []string) error {
	if len(args) < 1 || len(args) > 2 {
		return fmt.Errorf("использование: deduparc extract <архив.darc> [каталог]")
	}
	target := "."
	if len(args) == 2 {
		target = args[1]
	}
	r, err := archive.Open(args[0])
	if err != nil {
		return err
	}
	defer r.Close()
	if err := r.Extract(target); err != nil {
		return err
	}
	abs, _ := filepath.Abs(target)
	fmt.Println("Распаковано в:", abs)
	return nil
}

func commandVerify(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("использование: deduparc verify <архив.darc>")
	}
	r, err := archive.Open(args[0])
	if err != nil {
		return err
	}
	defer r.Close()
	result, err := r.Verify()
	if err != nil {
		return err
	}
	fmt.Printf("Архив исправен. Проверено файлов: %d\n", result.Files)
	return nil
}

func commandBridgeCreate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("использование: deduparc bridge-create <архив.darc> <каталог>")
	}
	root := filepath.Clean(args[1])
	items, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	inputs := make([]string, 0, len(items))
	for _, item := range items {
		inputs = append(inputs, filepath.Join(root, item.Name()))
	}
	_, err = archive.Create(args[0], inputs, archive.CreateOptions{})
	return err
}

func commandInfo(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("использование: deduparc info <архив.darc>")
	}
	r, err := archive.Open(args[0])
	if err != nil {
		return err
	}
	defer r.Close()
	s := r.Manifest.Stats
	fmt.Printf("Формат: %s/%d\n", r.Manifest.Format, r.Manifest.Version)
	fmt.Printf("Создан: %s\n", r.Manifest.CreatedAt)
	fmt.Printf("Файлов: %d\n", s.Files)
	fmt.Printf("Каталогов: %d\n", s.Directories)
	if s.SmallFiles > 0 {
		fmt.Printf("Мелких файлов: %d, групп: %d\n", s.SmallFiles, s.SmallPacks)
	}
	fmt.Printf("Исходный объём: %s\n", humanBytes(s.SourceBytes))
	if info, err := os.Stat(args[0]); err == nil {
		fmt.Printf("Размер архива: %s\n", humanBytes(info.Size()))
	}
	return nil
}

func ensureExtension(path string) string {
	if strings.EqualFold(filepath.Ext(path), ".darc") {
		return path
	}
	return path + ".darc"
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for value := n / unit; value >= unit && exp < 5; value /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func printHelp() {
	fmt.Printf(`DedupArc %s — небольшой файловый архиватор.

Использование:
  deduparc create [режим] <архив.darc> <файл|каталог> [...]  Создать архив
  deduparc list <архив.darc>                          Показать содержимое
  deduparc extract <архив.darc> [каталог]             Распаковать архив
  deduparc verify <архив.darc>                         Полностью проверить архив
  deduparc info <архив.darc>                           Показать статистику
  deduparc version                                     Показать версию

Режимы сжатия: --fast, --balanced (по умолчанию), --maximum, --extreme (экспериментальный)
Короткие команды: c, l, x, v, i
`, version.Current)
}
