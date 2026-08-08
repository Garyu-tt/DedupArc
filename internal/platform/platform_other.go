//go:build !windows

package platform

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func Message(title, text string)       { println(title + ": " + text) }
func AskYesNo(title, text string) bool { return true }
func ErrorMessage(title string, err error) {
	if err != nil {
		println(title + ": " + err.Error())
	}
}
func RunHidden(name string, args ...string) error   { return exec.Command(name, args...).Run() }
func RunElevated(name string, args ...string) error { return exec.Command(name, args...).Run() }
func StartHidden(name string, args ...string) error { return exec.Command(name, args...).Start() }
func InstallRoot() (string, error)                  { return "", errors.New("только Windows") }
func StateRoot() (string, error)                    { return filepath.Join(os.TempDir(), "DedupArc"), nil }
func Find7Zip() string                              { return "" }
func ProcessRunning(pid int) bool                   { return false }
func ProcessImageRunning(name string) bool          { return false }
func WaitForProcessesInPath(root string)            {}
func AssociationChanged()                           {}
func RegAdd(key, name, value string) error          { return errors.New("только Windows") }
func RegDelete(key string) error                    { return nil }
func QuoteCommand(path string, args ...string) string {
	return `"` + strings.ReplaceAll(path, `"`, `\"`) + `" "` + strings.Join(args, " ")
}
func CreateShortcut(shortcut, target, icon, workingDir string) error {
	return errors.New("только Windows")
}
func CurrentPIDString() string { return "0" }
