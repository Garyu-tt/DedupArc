package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/Garyu-tt/deduparc/internal/archive"
	"github.com/Garyu-tt/deduparc/internal/platform"
	"github.com/Garyu-tt/deduparc/internal/session"
	"github.com/Garyu-tt/deduparc/internal/version"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		platform.ErrorMessage("DedupArc", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) > 0 && strings.EqualFold(args[0], "--self-test") {
		return selfTest()
	}
	startUpdateCheck()
	if len(args) == 0 {
		return openManager("")
	}

	if len(args) == 1 && strings.EqualFold(filepath.Ext(args[0]), ".darc") {
		return session.OpenArchive(args[0])
	}

	switch strings.ToLower(args[0]) {
	case "--open":
		if len(args) != 2 {
			return errors.New("не указан архив")
		}
		return session.OpenArchive(args[1])
	case "--extract":
		if len(args) < 2 || len(args) > 3 {
			return errors.New("не указан архив")
		}
		target := ""
		if len(args) == 3 {
			target = args[2]
		}
		if err := session.ExtractArchive(args[1], target); err != nil {
			return err
		}
		platform.Message("DedupArc", "Архив распакован")
		return nil
	case "--verify":
		if len(args) != 2 {
			return errors.New("не указан архив")
		}
		r, err := archive.Open(args[1])
		if err != nil {
			return err
		}
		result, err := r.Verify()
		_ = r.Close()
		if err != nil {
			return err
		}
		platform.Message("DedupArc", fmt.Sprintf("Ошибок не найдено. Проверено файлов: %d", result.Files))
		return nil
	case "--create":
		if len(args) < 2 {
			return errors.New("не выбраны файлы или каталоги")
		}
		if err := session.CreateArchive("", args[1:]); err != nil {
			return err
		}
		platform.Message("DedupArc", "Архив создан")
		return nil
	case "--update":
		return launchUpdater(false)
	case "--version":
		platform.Message("DedupArc", version.Current)
		return nil
	default:
		if len(args) == 1 {
			return openManager(args[0])
		}
		return errors.New("неизвестная команда")
	}
}

func openManager(path string) error {
	fm := platform.Find7Zip()
	if fm == "" {
		return errors.New("7-Zip File Manager не найден. Запустите установщик DedupArc ещё раз")
	}
	args := []string{}
	if path != "" {
		args = append(args, path)
	}
	return exec.Command(fm, args...).Start()
}

func startUpdateCheck() {
	exe, err := os.Executable()
	if err != nil {
		return
	}
	updater := filepath.Join(filepath.Dir(exe), "DedupArcUpdater.exe")
	if _, err := os.Stat(updater); err != nil {
		return
	}
	_ = platform.StartHidden(updater, "--check-silent", platform.CurrentPIDString())
}

func launchUpdater(silent bool) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	updater := filepath.Join(filepath.Dir(exe), "DedupArcUpdater.exe")
	if _, err := os.Stat(updater); err != nil {
		return errors.New("модуль обновления не найден")
	}
	args := []string{"--check"}
	if silent {
		args = []string{"--check-silent"}
	}
	return platform.StartHidden(updater, args...)
}

func selfTest() error {
	if platform.Find7Zip() == "" {
		return errors.New("7-Zip File Manager не найден")
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	dir := filepath.Dir(exe)
	required := []string{"DedupArcUpdater.exe", "deduparc-core.exe", "DedupArc.ico"}
	for _, name := range required {
		if st, err := os.Stat(filepath.Join(dir, name)); err != nil || st.IsDir() {
			return fmt.Errorf("не найден файл %s", name)
		}
	}
	return selfTestRunnableSession()
}

func selfTestRunnableSession() error {
	root, err := os.MkdirTemp("", "DedupArc-self-test-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(root)

	app := filepath.Join(root, "Probe")
	if err := os.MkdirAll(filepath.Join(app, "data"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(app, "data", "ready.txt"), []byte("READY\r\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(app, "run.cmd"), []byte("@echo off\r\ntype data\\ready.txt\r\n"), 0o644); err != nil {
		return err
	}

	archivePath := filepath.Join(root, "probe.darc")
	if _, err := archive.Create(archivePath, []string{app}, archive.CreateOptions{}); err != nil {
		return fmt.Errorf("самопроверка создания архива: %w", err)
	}
	materialized := filepath.Join(root, "session")
	if err := session.MaterializeArchive(archivePath, materialized); err != nil {
		return fmt.Errorf("самопроверка рабочей сессии: %w", err)
	}

	cmd := exec.Command("cmd.exe", "/C", "run.cmd")
	cmd.Dir = filepath.Join(materialized, "Probe")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("самопроверка запуска из архива: %w", err)
	}
	if strings.TrimSpace(string(out)) != "READY" {
		return errors.New("самопроверка запуска из архива не получила ожидаемый ресурс")
	}
	return nil
}
