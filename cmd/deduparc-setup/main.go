package main

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Garyu-tt/deduparc/internal/platform"
	"github.com/Garyu-tt/deduparc/internal/updating"
	"github.com/Garyu-tt/deduparc/internal/version"
)

const sevenZipURL = "https://github.com/ip7z/7zip/releases/download/26.02/7z2602-x64.exe"
const sevenZipFallbackURL = "https://www.7-zip.org/a/7z2602-x64.exe"
const sevenZipSHA256 = "6745fa76dc2ea031596d8678f6f6b99c3c1b435b4164a63485adbbc7b8d82ef0"

//go:embed payload/*
var payload embed.FS

func main() {
	if err := run(os.Args[1:]); err != nil {
		platform.ErrorMessage("Установка DedupArc", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) > 0 && args[0] == "--uninstall" {
		return uninstall()
	}
	return install()
}

func install() error {
	if platform.ProcessImageRunning("DedupArc.exe") {
		return errors.New("закройте DedupArc перед установкой или обновлением")
	}
	if platform.Find7Zip() == "" {
		if !platform.AskYesNo("Установка DedupArc", "Для интерфейса DedupArc нужен 7-Zip File Manager. Установить официальный 7-Zip 26.02 автоматически?") {
			return errors.New("установка отменена")
		}
		if err := install7Zip(); err != nil {
			return err
		}
	}

	root, err := platform.InstallRoot()
	if err != nil {
		return err
	}
	parent := filepath.Dir(root)
	stage := filepath.Join(parent, ".DedupArc.install-"+strconv.FormatInt(time.Now().UnixNano(), 10))
	backup := filepath.Join(parent, ".DedupArc.previous")
	_ = os.RemoveAll(stage)
	if err := os.MkdirAll(stage, 0o755); err != nil {
		return err
	}
	if err := extractPayload(stage); err != nil {
		_ = os.RemoveAll(stage)
		return err
	}
	if err := copy7ZipRuntime(stage); err != nil {
		_ = os.RemoveAll(stage)
		return err
	}

	self, err := os.Executable()
	if err != nil {
		return err
	}
	if err := copyFile(self, filepath.Join(stage, "Uninstall.exe")); err != nil {
		return err
	}

	if err := updating.VerifyPayloadDir(stage); err != nil {
		_ = os.RemoveAll(stage)
		return fmt.Errorf("пакет установки повреждён: %w", err)
	}
	if oldConfig := filepath.Join(root, "config.json"); fileExists(oldConfig) {
		_ = copyFile(oldConfig, filepath.Join(stage, "config.json"))
	}

	_ = os.RemoveAll(backup)
	if fileExists(root) {
		if err := os.Rename(root, backup); err != nil {
			_ = os.RemoveAll(stage)
			return fmt.Errorf("не удалось заменить предыдущую версию: %w", err)
		}
	}
	if err := os.Rename(stage, root); err != nil {
		_ = os.Rename(backup, root)
		return err
	}

	if err := register(root); err != nil {
		rollback(root, backup)
		return err
	}
	if err := platform.RunHidden(filepath.Join(root, "DedupArc.exe"), "--self-test"); err != nil {
		rollback(root, backup)
		return fmt.Errorf("установка не прошла самопроверку: %w", err)
	}

	_ = os.RemoveAll(backup)
	platform.AssociationChanged()
	platform.Message("DedupArc", "DedupArc "+version.Current+" установлен. Файлы .darc теперь открываются двойным щелчком.")
	return platform.StartHidden(filepath.Join(root, "DedupArc.exe"))
}

func rollback(root, backup string) {
	failed := root + ".failed"
	_ = os.RemoveAll(failed)
	_ = os.Rename(root, failed)
	if fileExists(backup) {
		_ = os.Rename(backup, root)
	}
	_ = os.RemoveAll(failed)
}

func install7Zip() error {
	temp := filepath.Join(os.TempDir(), "7z2602-x64.exe")
	if err := downloadVerified(sevenZipURL, temp, sevenZipSHA256); err != nil {
		if fallbackErr := downloadVerified(sevenZipFallbackURL, temp, sevenZipSHA256); fallbackErr != nil {
			return fmt.Errorf("не удалось загрузить 7-Zip: %v; резервный источник: %w", err, fallbackErr)
		}
	}
	defer os.Remove(temp)
	if err := platform.RunElevated(temp, "/S"); err != nil {
		return fmt.Errorf("установщик 7-Zip завершился с ошибкой или elevation был отменён: %w", err)
	}
	for i := 0; i < 20; i++ {
		if platform.Find7Zip() != "" {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return errors.New("7-Zip установился, но 7zFM.exe не найден")
}

func downloadVerified(url, path, want string) error {
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("сервер вернул %s", resp.Status)
	}
	tmp := path + ".part"
	_ = os.Remove(tmp)
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	h := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(f, h), resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if !strings.EqualFold(hex.EncodeToString(h.Sum(nil)), want) {
		_ = os.Remove(tmp)
		return errors.New("SHA-256 загруженного 7-Zip не совпадает")
	}
	_ = os.Remove(path)
	return os.Rename(tmp, path)
}

func extractPayload(target string) error {
	return fs.WalkDir(payload, "payload", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == "payload" {
			return nil
		}
		rel := strings.TrimPrefix(filepath.ToSlash(path), "payload/")
		dst := filepath.Join(target, filepath.FromSlash(rel))
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		b, err := payload.ReadFile(path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		return os.WriteFile(dst, b, 0o755)
	})
}

func copy7ZipRuntime(stage string) error {
	fm := platform.Find7Zip()
	if fm == "" {
		return errors.New("7-Zip File Manager не найден")
	}
	src := filepath.Dir(fm)
	dst := filepath.Join(stage, "runtime", "7zip")
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return os.MkdirAll(dst, 0o755)
		}
		name := strings.ToLower(d.Name())
		if !d.IsDir() && (name == "uninstall.exe" || name == "uninstall.exe.manifest") {
			return nil
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		return copyFile(path, target)
	})
}

func register(root string) error {
	exe := filepath.Join(root, "DedupArc.exe")
	ico := filepath.Join(root, "DedupArc.ico")
	uninstaller := filepath.Join(root, "Uninstall.exe")
	base := `HKCU\Software\Classes\DedupArc.Archive`
	entries := [][3]string{
		{`HKCU\Software\Classes\.darc`, "", "DedupArc.Archive"},
		{base, "", "Архив DedupArc"},
		{base + `\DefaultIcon`, "", ico},
		{base + `\shell\open`, "", "Открыть"},
		{base + `\shell\open\command`, "", platform.QuoteCommand(exe, `"%1"`)},
		{base + `\shell\extract`, "", "Извлечь сюда"},
		{base + `\shell\extract\command`, "", platform.QuoteCommand(exe, "--extract", `"%1"`)},
		{base + `\shell\verify`, "", "Проверить архив"},
		{base + `\shell\verify\command`, "", platform.QuoteCommand(exe, "--verify", `"%1"`)},
		{`HKCU\Software\Classes\*\shell\DedupArc`, "", "Добавить в архив DedupArc"},
		{`HKCU\Software\Classes\*\shell\DedupArc`, "Icon", ico},
		{`HKCU\Software\Classes\*\shell\DedupArc\command`, "", platform.QuoteCommand(exe, "--create", `"%1"`)},
		{`HKCU\Software\Classes\Directory\shell\DedupArc`, "", "Добавить в архив DedupArc"},
		{`HKCU\Software\Classes\Directory\shell\DedupArc`, "Icon", ico},
		{`HKCU\Software\Classes\Directory\shell\DedupArc\command`, "", platform.QuoteCommand(exe, "--create", `"%1"`)},
	}
	for _, e := range entries {
		if err := platform.RegAdd(e[0], e[1], e[2]); err != nil {
			return err
		}
	}

	uninstallKey := `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\DedupArc`
	fields := map[string]string{"DisplayName": "DedupArc", "DisplayVersion": version.Current, "Publisher": "Garyu-tt", "DisplayIcon": ico, "InstallLocation": root, "UninstallString": platform.QuoteCommand(uninstaller, "--uninstall")}
	for name, value := range fields {
		if err := platform.RegAdd(uninstallKey, name, value); err != nil {
			return err
		}
	}

	appdata := os.Getenv("APPDATA")
	if appdata != "" {
		shortcut := filepath.Join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "DedupArc.lnk")
		_ = os.MkdirAll(filepath.Dir(shortcut), 0o755)
		_ = platform.CreateShortcut(shortcut, exe, ico, root)
	}
	return nil
}

func uninstall() error {
	root, err := platform.InstallRoot()
	if err != nil {
		return err
	}
	if platform.ProcessImageRunning("DedupArc.exe") {
		return errors.New("закройте DedupArc перед удалением")
	}
	_ = platform.RegDelete(`HKCU\Software\Classes\.darc`)
	_ = platform.RegDelete(`HKCU\Software\Classes\DedupArc.Archive`)
	_ = platform.RegDelete(`HKCU\Software\Classes\*\shell\DedupArc`)
	_ = platform.RegDelete(`HKCU\Software\Classes\Directory\shell\DedupArc`)
	_ = platform.RegDelete(`HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\DedupArc`)
	if appdata := os.Getenv("APPDATA"); appdata != "" {
		_ = os.Remove(filepath.Join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "DedupArc.lnk"))
	}
	platform.AssociationChanged()

	self, _ := os.Executable()
	script := filepath.Join(os.TempDir(), "deduparc-remove-"+strconv.FormatInt(time.Now().UnixNano(), 10)+".cmd")
	body := "@echo off\r\nping 127.0.0.1 -n 3 >nul\r\nrmdir /s /q \"" + root + "\"\r\ndel /q \"" + self + "\" 2>nul\r\ndel /q \"%~f0\"\r\n"
	if err := os.WriteFile(script, []byte(body), 0o700); err != nil {
		return err
	}
	platform.Message("DedupArc", "DedupArc удалён. Пользовательские архивы не затронуты.")
	return platform.StartHidden("cmd.exe", "/c", script)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	cerr := out.Close()
	if err != nil {
		return err
	}
	return cerr
}
func fileExists(path string) bool { _, err := os.Stat(path); return err == nil }
