//go:build !windows

package platform

func WithProgress(title string, task func(ProgressUpdateFunc) error) error {
	return task(func(ProgressUpdate) bool { return true })
}
