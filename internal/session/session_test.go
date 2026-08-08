package session

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/Garyu-tt/deduparc/internal/archive"
)

func TestMaterializedArchiveKeepsRunnableWorkingSet(t *testing.T) {
	root := t.TempDir()
	game := filepath.Join(root, "MiniGame")
	data := filepath.Join(game, "data")
	if err := os.MkdirAll(data, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(data, "config.txt"), []byte("READY\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	launcherName := "run.sh"
	launcherBody := "#!/bin/sh\ncat data/config.txt\n"
	if runtime.GOOS == "windows" {
		launcherName = "run.cmd"
		launcherBody = "@echo off\r\ntype data\\config.txt\r\n"
	}
	launcher := filepath.Join(game, launcherName)
	if err := os.WriteFile(launcher, []byte(launcherBody), 0o755); err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(root, "game.darc")
	if _, err := archive.Create(archivePath, []string{game}, archive.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	sessionRoot := filepath.Join(root, "session")
	if err := MaterializeArchive(archivePath, sessionRoot); err != nil {
		t.Fatal(err)
	}

	workDir := filepath.Join(sessionRoot, "MiniGame")
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd.exe", "/C", launcherName)
	} else {
		cmd = exec.Command("sh", launcherName)
	}
	cmd.Dir = workDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("запуск из рабочей сессии завершился ошибкой: %v\n%s", err, out)
	}
	if strings.TrimSpace(string(out)) != "READY" {
		t.Fatalf("программа не увидела соседний ресурс: %q", out)
	}
}
