//go:build windows

package platform

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

var (
	user32             = syscall.NewLazyDLL("user32.dll")
	shell32            = syscall.NewLazyDLL("shell32.dll")
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	procMessageBoxW    = user32.NewProc("MessageBoxW")
	procSHChangeNotify = shell32.NewProc("SHChangeNotify")
	procOpenProcess    = kernel32.NewProc("OpenProcess")
	procCloseHandle    = kernel32.NewProc("CloseHandle")
)

const (
	mbOK                           = 0x00000000
	mbIconError                    = 0x00000010
	mbIconInformation              = 0x00000040
	mbYesNo                        = 0x00000004
	idYes                          = 6
	processQueryLimitedInformation = 0x1000
	shcneAssocChanged              = 0x08000000
)

func utf16ptr(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func Message(title, text string) {
	procMessageBoxW.Call(0, uintptr(unsafe.Pointer(utf16ptr(text))), uintptr(unsafe.Pointer(utf16ptr(title))), mbOK|mbIconInformation)
}

func AskYesNo(title, text string) bool {
	r, _, _ := procMessageBoxW.Call(0, uintptr(unsafe.Pointer(utf16ptr(text))), uintptr(unsafe.Pointer(utf16ptr(title))), mbYesNo|mbIconInformation)
	return r == idYes
}

func ErrorMessage(title string, err error) {
	text := "Неизвестная ошибка"
	if err != nil {
		text = err.Error()
	}
	procMessageBoxW.Call(0, uintptr(unsafe.Pointer(utf16ptr(text))), uintptr(unsafe.Pointer(utf16ptr(title))), mbOK|mbIconError)
}

func RunHidden(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg != "" {
			return fmt.Errorf("%s: %w", msg, err)
		}
		return err
	}
	return nil
}

func RunElevated(name string, args ...string) error {
	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		quoted = append(quoted, "'"+psEscape(arg)+"'")
	}
	script := `$p=Start-Process -FilePath '` + psEscape(name) + `' -ArgumentList @(` + strings.Join(quoted, ",") + `) -Verb RunAs -Wait -PassThru; if ($p.ExitCode -ne 0) { exit $p.ExitCode }`
	return RunHidden("powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script)
}

func StartHidden(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	return cmd.Start()
}

func InstallRoot() (string, error) {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		return "", errors.New("переменная LOCALAPPDATA не задана")
	}
	return filepath.Join(base, "Programs", "DedupArc"), nil
}

func StateRoot() (string, error) {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		return "", errors.New("переменная LOCALAPPDATA не задана")
	}
	return filepath.Join(base, "DedupArc"), nil
}

func Find7Zip() string {
	candidates := []string{}
	if pf := os.Getenv("ProgramFiles"); pf != "" {
		candidates = append(candidates, filepath.Join(pf, "7-Zip", "7zFM.exe"))
	}
	if pfx86 := os.Getenv("ProgramFiles(x86)"); pfx86 != "" {
		candidates = append(candidates, filepath.Join(pfx86, "7-Zip", "7zFM.exe"))
	}
	if root, err := InstallRoot(); err == nil {
		candidates = append([]string{filepath.Join(root, "runtime", "7zip", "7zFM.exe")}, candidates...)
	}
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}
	return ""
}

func ProcessRunning(pid int) bool {
	if pid <= 0 {
		return false
	}
	h, _, _ := procOpenProcess.Call(processQueryLimitedInformation, 0, uintptr(pid))
	if h == 0 {
		return false
	}
	procCloseHandle.Call(h)
	return true
}

func ProcessImageRunning(name string) bool {
	cmd := exec.Command("tasklist.exe", "/FI", "IMAGENAME eq "+name, "/NH")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	text := strings.ToLower(string(out))
	return strings.Contains(text, strings.ToLower(name))
}

func WaitForProcessesInPath(root string) {
	clean, err := filepath.Abs(root)
	if err != nil {
		return
	}
	clean = strings.TrimRight(clean, `\/`) + `\`
	script := `$root='` + psEscape(clean) + `'; $found=Get-Process -ErrorAction SilentlyContinue | Where-Object { try { $p=$_.Path; $p -and $p.StartsWith($root,[System.StringComparison]::OrdinalIgnoreCase) } catch { $false } } | Select-Object -First 1; if ($found) { exit 1 } else { exit 0 }`
	for {
		cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script)
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
		if err := cmd.Run(); err == nil {
			return
		}
		time.Sleep(900 * time.Millisecond)
	}
}

func AssociationChanged() { procSHChangeNotify.Call(shcneAssocChanged, 0, 0, 0) }

func RegAdd(key, name, value string) error {
	args := []string{"ADD", key, "/f"}
	if name == "" {
		args = append(args, "/ve")
	} else {
		args = append(args, "/v", name)
	}
	args = append(args, "/t", "REG_SZ", "/d", value)
	return RunHidden("reg.exe", args...)
}

func RegDelete(key string) error {
	cmd := exec.Command("reg.exe", "DELETE", key, "/f")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	_ = cmd.Run()
	return nil
}

func QuoteCommand(path string, args ...string) string {
	q := `"` + strings.ReplaceAll(path, `"`, `\"`) + `"`
	if len(args) == 0 {
		return q
	}
	return q + " " + strings.Join(args, " ")
}

func CreateShortcut(shortcut, target, icon, workingDir string) error {
	script := `$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut('` + psEscape(shortcut) + `'); $s.TargetPath='` + psEscape(target) + `'; $s.WorkingDirectory='` + psEscape(workingDir) + `'; $s.IconLocation='` + psEscape(icon) + `'; $s.Save()`
	return RunHidden("powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script)
}

func psEscape(s string) string { return strings.ReplaceAll(s, "'", "''") }

func CurrentPIDString() string { return strconv.Itoa(os.Getpid()) }
