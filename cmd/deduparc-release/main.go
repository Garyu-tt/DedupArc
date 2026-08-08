package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Garyu-tt/deduparc/internal/updating"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}
func run(args []string) error {
	if len(args) < 4 || args[0] != "manifest" {
		return fmt.Errorf("использование: deduparc-release manifest <version> <url> <package.zip> [notes]")
	}
	data, err := os.ReadFile(args[3])
	if err != nil {
		return err
	}
	sum := sha256.Sum256(data)
	notes := ""
	if len(args) > 4 {
		notes = args[4]
	}
	manifest, err := updating.BuildManifest(args[1], args[2], hex.EncodeToString(sum[:]), int64(len(data)), notes)
	if err != nil {
		return err
	}
	key := os.Getenv("DEDUPARC_RELEASE_KEY")
	if key == "" {
		return fmt.Errorf("не задан DEDUPARC_RELEASE_KEY")
	}
	sig, err := updating.SignManifest(manifest, key)
	if err != nil {
		return err
	}
	dir := filepath.Dir(args[3])
	if err := os.WriteFile(filepath.Join(dir, "update.json"), manifest, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "update.sig"), []byte(sig+"\n"), 0o644)
}
