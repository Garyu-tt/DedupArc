package updating

import (
	"archive/zip"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Garyu-tt/deduparc/internal/platform"
	"github.com/Garyu-tt/deduparc/internal/version"
)

const (
	CurrentVersion  = version.Current
	ManifestURL     = "https://github.com/Garyu-tt/DedupArc/releases/latest/download/update.json"
	SignatureURL    = "https://github.com/Garyu-tt/DedupArc/releases/latest/download/update.sig"
	publicKeyBase64 = "wusuYSwCq3mTIECOHanWVwHQNUBrv3yfxMngXD9F+Og="
)

type Manifest struct {
	Version string `json:"version"`
	URL     string `json:"url"`
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size,omitempty"`
	Notes   string `json:"notes,omitempty"`
}

type CheckState struct {
	LastCheck int64 `json:"last_check"`
}

func Check(silent bool, parentPID int) error {
	if silent {
		due, err := checkDue()
		if err == nil && !due {
			return nil
		}
	}
	manifestBytes, sigBytes, err := fetchManifest()
	recordCheck()
	if err != nil {
		if silent {
			return nil
		}
		return err
	}
	m, err := verifyManifest(manifestBytes, sigBytes)
	if err != nil {
		return err
	}
	if compareVersions(m.Version, CurrentVersion) <= 0 {
		if !silent {
			platform.Message("DedupArc", "Установлена последняя версия")
		}
		return nil
	}
	text := "Доступна версия " + m.Version + ". Установить обновление?"
	if m.Notes != "" {
		text += "\n\n" + m.Notes
	}
	if !platform.AskYesNo("Обновление DedupArc", text) {
		return nil
	}

	zipPath, err := downloadPackage(m)
	if err != nil {
		return err
	}
	return launchApplier(zipPath, m.Version, parentPID)
}

func Apply(zipPath, installRoot, version string, parentPID int) error {
	deadline := time.Now().Add(10 * time.Minute)
	for {
		parentAlive := parentPID > 0 && platform.ProcessRunning(parentPID)
		appAlive := platform.ProcessImageRunning("DedupArc.exe")
		if !parentAlive && !appAlive {
			break
		}
		if time.Now().After(deadline) {
			return errors.New("обновление отменено: DedupArc всё ещё запущен")
		}
		time.Sleep(750 * time.Millisecond)
	}

	parentDir := filepath.Dir(installRoot)
	stage := filepath.Join(parentDir, ".DedupArc.update-"+strconv.FormatInt(time.Now().UnixNano(), 10))
	backup := filepath.Join(parentDir, ".DedupArc.backup")
	_ = os.RemoveAll(stage)
	if err := os.MkdirAll(stage, 0o755); err != nil {
		return err
	}
	if err := extractZip(zipPath, stage); err != nil {
		_ = os.RemoveAll(stage)
		return err
	}
	if err := verifyPayload(stage); err != nil {
		_ = os.RemoveAll(stage)
		return err
	}

	if oldConfig := filepath.Join(installRoot, "config.json"); fileExists(oldConfig) {
		_ = copyFile(oldConfig, filepath.Join(stage, "config.json"))
	}
	if oldRuntime := filepath.Join(installRoot, "runtime"); fileExists(oldRuntime) {
		if err := copyDir(oldRuntime, filepath.Join(stage, "runtime")); err != nil {
			_ = os.RemoveAll(stage)
			return fmt.Errorf("не удалось сохранить runtime: %w", err)
		}
	}

	_ = os.RemoveAll(backup)
	if fileExists(installRoot) {
		if err := os.Rename(installRoot, backup); err != nil {
			_ = os.RemoveAll(stage)
			return fmt.Errorf("не удалось подготовить замену текущей версии: %w", err)
		}
	}
	if err := os.Rename(stage, installRoot); err != nil {
		_ = os.Rename(backup, installRoot)
		return fmt.Errorf("не удалось активировать новую версию: %w", err)
	}

	newExe := filepath.Join(installRoot, "DedupArc.exe")
	if err := platform.RunHidden(newExe, "--self-test"); err != nil {
		broken := filepath.Join(parentDir, ".DedupArc.failed-"+version)
		_ = os.RemoveAll(broken)
		_ = os.Rename(installRoot, broken)
		_ = os.Rename(backup, installRoot)
		_ = os.RemoveAll(broken)
		return fmt.Errorf("новая версия не прошла самопроверку, выполнен откат: %w", err)
	}

	_ = os.RemoveAll(backup)
	_ = os.Remove(zipPath)
	platform.AssociationChanged()
	platform.Message("DedupArc", "Обновление до версии "+version+" установлено")
	return platform.StartHidden(newExe)
}

func fetchManifest() ([]byte, []byte, error) {
	client := &http.Client{Timeout: 20 * time.Second}
	mb, err := get(client, ManifestURL)
	if err != nil {
		return nil, nil, err
	}
	sb, err := get(client, SignatureURL)
	if err != nil {
		return nil, nil, err
	}
	return mb, sb, nil
}

func verifyManifest(data, sigData []byte) (Manifest, error) {
	var m Manifest
	pubRaw, err := base64.StdEncoding.DecodeString(publicKeyBase64)
	if err != nil {
		return m, err
	}
	sigText := strings.TrimSpace(string(sigData))
	sig, err := base64.StdEncoding.DecodeString(sigText)
	if err != nil {
		return m, errors.New("подпись обновления повреждена")
	}
	if !ed25519.Verify(ed25519.PublicKey(pubRaw), data, sig) {
		return m, errors.New("подпись обновления не прошла проверку")
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return m, err
	}
	if m.Version == "" || m.URL == "" || len(m.SHA256) != 64 {
		return m, errors.New("некорректный манифест обновления")
	}
	return m, nil
}

func downloadPackage(m Manifest) (string, error) {
	state, err := platform.StateRoot()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(state, "Updates")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "DedupArc-"+m.Version+".zip")
	tmp := path + ".part"
	_ = os.Remove(tmp)

	client := &http.Client{Timeout: 15 * time.Minute}
	resp, err := client.Get(m.URL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("сервер обновлений вернул %s", resp.Status)
	}
	f, err := os.Create(tmp)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(f, h), resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return "", copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return "", closeErr
	}
	if m.Size > 0 && n != m.Size {
		_ = os.Remove(tmp)
		return "", errors.New("размер пакета обновления не совпадает")
	}
	if !strings.EqualFold(hex.EncodeToString(h.Sum(nil)), m.SHA256) {
		_ = os.Remove(tmp)
		return "", errors.New("SHA-256 пакета обновления не совпадает")
	}
	_ = os.Remove(path)
	if err := os.Rename(tmp, path); err != nil {
		return "", err
	}
	return path, nil
}

func launchApplier(zipPath, version string, parentPID int) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	root, err := platform.InstallRoot()
	if err != nil {
		return err
	}
	temp := filepath.Join(os.TempDir(), "DedupArcUpdater-"+strconv.FormatInt(time.Now().UnixNano(), 10)+".exe")
	if err := copyFile(exe, temp); err != nil {
		return err
	}
	return platform.StartHidden(temp, "--apply", zipPath, root, version, strconv.Itoa(parentPID))
}

func get(client *http.Client, url string) ([]byte, error) {
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("сервер обновлений вернул %s", resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 4<<20))
}

func checkDue() (bool, error) {
	path, err := stateFile()
	if err != nil {
		return true, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return true, nil
	}
	var s CheckState
	if json.Unmarshal(b, &s) != nil {
		return true, nil
	}
	return time.Since(time.Unix(s.LastCheck, 0)) >= 24*time.Hour, nil
}

func recordCheck() {
	path, err := stateFile()
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	b, _ := json.Marshal(CheckState{LastCheck: time.Now().Unix()})
	_ = os.WriteFile(path, b, 0o600)
}

func stateFile() (string, error) {
	root, err := platform.StateRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "updater.json"), nil
}

func compareVersions(a, b string) int {
	pa := parseVersion(a)
	pb := parseVersion(b)
	for i := 0; i < 3; i++ {
		if pa[i] < pb[i] {
			return -1
		}
		if pa[i] > pb[i] {
			return 1
		}
	}
	return 0
}
func parseVersion(v string) [3]int {
	var out [3]int
	parts := strings.Split(strings.TrimPrefix(v, "v"), ".")
	for i := 0; i < len(parts) && i < 3; i++ {
		out[i], _ = strconv.Atoi(parts[i])
	}
	return out
}

func extractZip(zipPath, target string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()
	base, err := filepath.Abs(target)
	if err != nil {
		return err
	}
	for _, f := range r.File {
		clean := filepath.Clean(filepath.FromSlash(f.Name))
		if clean == "." || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("небезопасный путь в пакете обновления")
		}
		dst := filepath.Join(base, clean)
		rel, err := filepath.Rel(base, dst)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return errors.New("путь пакета выходит за каталог установки")
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(dst, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		src, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, f.Mode())
		if err != nil {
			src.Close()
			return err
		}
		_, cErr := io.Copy(out, src)
		sErr := src.Close()
		oErr := out.Close()
		if cErr != nil {
			return cErr
		}
		if sErr != nil {
			return sErr
		}
		if oErr != nil {
			return oErr
		}
	}
	return nil
}

type payloadManifest struct {
	Files map[string]string `json:"files"`
}

func verifyPayload(root string) error {
	b, err := os.ReadFile(filepath.Join(root, "payload-manifest.json"))
	if err != nil {
		return err
	}
	var m payloadManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	if len(m.Files) == 0 {
		return errors.New("пустой список файлов обновления")
	}
	for rel, expected := range m.Files {
		clean := filepath.Clean(filepath.FromSlash(rel))
		if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("некорректный путь в списке файлов")
		}
		data, err := os.ReadFile(filepath.Join(root, clean))
		if err != nil {
			return fmt.Errorf("не найден %s", rel)
		}
		sum := sha256.Sum256(data)
		if !strings.EqualFold(hex.EncodeToString(sum[:]), expected) {
			return fmt.Errorf("контрольная сумма %s не совпадает", rel)
		}
	}
	return nil
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
	cErr := out.Close()
	if err != nil {
		return err
	}
	return cErr
}
func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		return copyFile(path, target)
	})
}

func fileExists(path string) bool { _, err := os.Stat(path); return err == nil }

// SignManifest используется release-скриптом и тестами.
func SignManifest(data []byte, privateKeyBase64 string) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(privateKeyBase64))
	if err != nil {
		return "", err
	}
	var key ed25519.PrivateKey
	switch len(raw) {
	case ed25519.SeedSize:
		key = ed25519.NewKeyFromSeed(raw)
	case ed25519.PrivateKeySize:
		key = ed25519.PrivateKey(raw)
	default:
		return "", errors.New("некорректный ключ подписи")
	}
	sig := ed25519.Sign(key, data)
	return base64.StdEncoding.EncodeToString(sig), nil
}

func BuildManifest(version, url, sha string, size int64, notes string) ([]byte, error) {
	m := Manifest{Version: version, URL: url, SHA256: sha, Size: size, Notes: notes}
	return json.MarshalIndent(m, "", "  ")
}

func VerifyManifestForTest(data []byte, sig string) error {
	_, err := verifyManifest(data, []byte(sig))
	return err
}

// VerifyPayloadDir проверяет внутренний список контрольных сумм установленного пакета.
func VerifyPayloadDir(root string) error { return verifyPayload(root) }
