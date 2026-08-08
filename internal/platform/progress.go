package platform

type ProgressUpdate struct {
	Stage   string
	Detail  string
	Current int64
	Total   int64
}

type ProgressUpdateFunc func(ProgressUpdate) bool
