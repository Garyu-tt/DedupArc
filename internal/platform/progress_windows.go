//go:build windows

package platform

import (
	"runtime"
	"syscall"
	"unsafe"
)

type guid struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

type progressDialog struct {
	vtbl *progressDialogVtbl
}

type progressDialogVtbl struct {
	queryInterface      uintptr
	addRef              uintptr
	release             uintptr
	startProgressDialog uintptr
	stopProgressDialog  uintptr
	setTitle            uintptr
	setAnimation        uintptr
	hasUserCancelled    uintptr
	setProgress         uintptr
	setProgress64       uintptr
	setLine             uintptr
	setCancelMsg        uintptr
	timer               uintptr
}

var (
	ole32                = syscall.NewLazyDLL("ole32.dll")
	procCoInitializeEx   = ole32.NewProc("CoInitializeEx")
	procCoUninitialize   = ole32.NewProc("CoUninitialize")
	procCoCreateInstance = ole32.NewProc("CoCreateInstance")
	clsidProgressDialog  = guid{0xF8383852, 0xFCD3, 0x11D1, [8]byte{0xA6, 0xB9, 0x00, 0x60, 0x97, 0xDF, 0x5B, 0xD4}}
	iidIProgressDialog   = guid{0xEBBC7C04, 0x315E, 0x11D2, [8]byte{0xB6, 0x2F, 0x00, 0x60, 0x97, 0xDF, 0x5B, 0xD4}}
)

const (
	coinitApartmentThreaded = 0x2
	clsctxInprocServer      = 0x1
	progdlgAutoTime         = 0x2
	progdlgNoMinimize       = 0x8
)

func WithProgress(title string, task func(ProgressUpdateFunc) error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	hr, _, _ := procCoInitializeEx.Call(0, coinitApartmentThreaded)
	initialized := hresultSucceeded(hr)
	if initialized {
		defer procCoUninitialize.Call()
	}

	var dialog *progressDialog
	createHR, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&clsidProgressDialog)),
		0,
		clsctxInprocServer,
		uintptr(unsafe.Pointer(&iidIProgressDialog)),
		uintptr(unsafe.Pointer(&dialog)),
	)
	if !hresultSucceeded(createHR) || dialog == nil || dialog.vtbl == nil {
		return task(func(ProgressUpdate) bool { return true })
	}
	defer syscall.SyscallN(dialog.vtbl.release, uintptr(unsafe.Pointer(dialog)))

	setProgressTitle(dialog, title)
	syscall.SyscallN(
		dialog.vtbl.startProgressDialog,
		uintptr(unsafe.Pointer(dialog)),
		0,
		0,
		progdlgAutoTime|progdlgNoMinimize,
		0,
	)
	defer syscall.SyscallN(dialog.vtbl.stopProgressDialog, uintptr(unsafe.Pointer(dialog)))
	setProgressLine(dialog, 1, "Подготовка...")
	setProgressLine(dialog, 2, "")

	update := func(p ProgressUpdate) bool {
		if userCancelled(dialog) {
			setProgressCancelMessage(dialog, "Операция отменяется...")
			return false
		}
		if p.Stage != "" {
			setProgressLine(dialog, 1, p.Stage)
		}
		setProgressLine(dialog, 2, p.Detail)
		total := p.Total
		if total <= 0 {
			total = 1
		}
		current := p.Current
		if current < 0 {
			current = 0
		}
		if current > total {
			current = total
		}
		syscall.SyscallN(
			dialog.vtbl.setProgress64,
			uintptr(unsafe.Pointer(dialog)),
			uintptr(current),
			uintptr(total),
		)
		return !userCancelled(dialog)
	}

	return task(update)
}

func hresultSucceeded(hr uintptr) bool {
	return int32(hr) >= 0
}

func userCancelled(dialog *progressDialog) bool {
	r, _, _ := syscall.SyscallN(dialog.vtbl.hasUserCancelled, uintptr(unsafe.Pointer(dialog)))
	return r != 0
}

func setProgressTitle(dialog *progressDialog, text string) {
	p, _ := syscall.UTF16PtrFromString(text)
	syscall.SyscallN(dialog.vtbl.setTitle, uintptr(unsafe.Pointer(dialog)), uintptr(unsafe.Pointer(p)))
}

func setProgressLine(dialog *progressDialog, line uintptr, text string) {
	p, _ := syscall.UTF16PtrFromString(text)
	syscall.SyscallN(
		dialog.vtbl.setLine,
		uintptr(unsafe.Pointer(dialog)),
		line,
		uintptr(unsafe.Pointer(p)),
		0,
		0,
	)
}

func setProgressCancelMessage(dialog *progressDialog, text string) {
	p, _ := syscall.UTF16PtrFromString(text)
	syscall.SyscallN(dialog.vtbl.setCancelMsg, uintptr(unsafe.Pointer(dialog)), uintptr(unsafe.Pointer(p)), 0)
}
