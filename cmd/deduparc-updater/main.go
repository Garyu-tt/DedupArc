package main

import (
	"errors"
	"os"
	"strconv"

	"github.com/Garyu-tt/deduparc/internal/platform"
	"github.com/Garyu-tt/deduparc/internal/updating"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		platform.ErrorMessage("Обновление DedupArc", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return updating.Check(false, 0)
	}
	switch args[0] {
	case "--check":
		parent := argPID(args, 1)
		return updating.Check(false, parent)
	case "--check-silent":
		parent := argPID(args, 1)
		return updating.Check(true, parent)
	case "--apply":
		if len(args) != 5 {
			return errors.New("некорректные параметры применения обновления")
		}
		pid, _ := strconv.Atoi(args[4])
		return updating.Apply(args[1], args[2], args[3], pid)
	default:
		return errors.New("неизвестная команда обновлятора")
	}
}

func argPID(args []string, index int) int {
	if len(args) <= index {
		return 0
	}
	pid, _ := strconv.Atoi(args[index])
	return pid
}
