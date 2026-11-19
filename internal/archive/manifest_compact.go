package archive

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"unicode/utf8"
)

// An optional v6 manifest serialization: pack only inline-file metadata into
// a length-delimited binary table. The entire manifest is still SHA-256 integrity-checked (not signed)
// and compressed by MNF2, and public Manifest stays unchanged.
// Always choose it only if it actually reduces the stored manifest.
const inlineTableMaxFiles = 1_000_000

func marshalManifestV6(manifest *Manifest) ([]byte,error) {
	plain,err:=json.Marshal(manifest)
	if err!=nil || manifest.Version!=6 {return plain,err}
	inline:=0
	for _,e:=range manifest.Entries {if len(e.InlineMath)>0 {inline++}}
	if inline==0 {return plain,nil}

	table:=binary.AppendUvarint(nil,uint64(inline))
	ordinary:=make([]Entry,0,len(manifest.Entries)-inline)
	for _,e:=range manifest.Entries {
		if len(e.InlineMath)==0 {
			ordinary=append(ordinary,e)
			continue
		}
		hash,err:=hex.DecodeString(e.SHA256)
		if err!=nil || len(hash)!=32 {return nil,fmt.Errorf("inline table: invalid SHA-256")}
		table=binary.AppendUvarint(table,uint64(len(e.Path)))
		table=append(table,e.Path...)
		table=binary.AppendUvarint(table,uint64(e.Size))
		table=binary.AppendUvarint(table,uint64(e.Mode))
		table=binary.AppendVarint(table,e.ModifiedUnix)
		table=append(table,hash...)
		table=binary.AppendUvarint(table,uint64(len(e.InlineMath)))
		table=append(table,e.InlineMath...)
	}
	compact:=*manifest
	compact.Entries=ordinary
	compact.InlineTable=table
	candidate,err:=json.Marshal(&compact)
	if err!=nil {return nil,err}
	a,err:=zlibCompress(plain,9)
	if err!=nil {return nil,err}
	b,err:=zlibCompress(candidate,9)
	if err!=nil {return nil,err}
	if len(b)<len(a) {return candidate,nil}
	return plain,nil
}

type inlineTableCursor struct {
	data []byte
	at int
}
func (c *inlineTableCursor) u() (uint64,error) {
	if c.at>=len(c.data) {return 0,ErrInvalidArchive}
	v,n:=binary.Uvarint(c.data[c.at:]);if n<=0 {return 0,ErrInvalidArchive}
	c.at+=n;return v,nil
}
func (c *inlineTableCursor) s() (int64,error) {
	if c.at>=len(c.data) {return 0,ErrInvalidArchive}
	v,n:=binary.Varint(c.data[c.at:]);if n<=0 {return 0,ErrInvalidArchive}
	c.at+=n;return v,nil
}
func (c *inlineTableCursor) bytes(n uint64) ([]byte,error) {
	if n>uint64(len(c.data)-c.at) {return nil,ErrInvalidArchive}
	out:=c.data[c.at:c.at+int(n)]
	c.at+=int(n)
	return out,nil
}

func expandInlineTable(manifest *Manifest) error {
	if len(manifest.InlineTable)==0 {return nil}
	if manifest.Version!=6 {return ErrInvalidArchive}
	c:=inlineTableCursor{data:manifest.InlineTable}
	count,err:=c.u()
	if err!=nil || count>inlineTableMaxFiles || count>uint64(len(c.data)/40+1) {
		return ErrInvalidArchive
	}
	for i:=uint64(0);i<count;i++ {
		pathLen,err:=c.u();if err!=nil || pathLen==0 || pathLen>1<<16 {return ErrInvalidArchive}
		path,err:=c.bytes(pathLen);if err!=nil || !utf8.Valid(path) {return ErrInvalidArchive}
		size,err:=c.u();if err!=nil || size<inlineMathMinRaw || size>inlineMathMaxRaw {return ErrInvalidArchive}
		mode,err:=c.u();if err!=nil || mode>uint64(^uint32(0)) {return ErrInvalidArchive}
		mtime,err:=c.s();if err!=nil {return ErrInvalidArchive}
		hash,err:=c.bytes(32);if err!=nil {return ErrInvalidArchive}
		codeLen,err:=c.u();if err!=nil || codeLen<2 || codeLen>inlineMathMaxCode {return ErrInvalidArchive}
		code,err:=c.bytes(codeLen);if err!=nil {return ErrInvalidArchive}
		manifest.Entries=append(manifest.Entries,Entry{
			Path:string(path), Type:"file", Size:int64(size), Mode:uint32(mode),
			ModifiedUnix:mtime, SHA256:hex.EncodeToString(hash),
			InlineMath:append([]byte(nil),code...),
		})
	}
	if c.at!=len(c.data) {return ErrInvalidArchive}
	// Canonical directory order is preserved even though the compact
	// entries were stored separately from ordinary v6 file entries.
	sort.Slice(manifest.Entries,func(i,j int)bool{return manifest.Entries[i].Path<manifest.Entries[j].Path})
	manifest.InlineTable=nil
	return nil
}
