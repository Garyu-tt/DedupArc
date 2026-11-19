package archive

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/Garyu-tt/deduparc/internal/chunker"
)

func TestNeutronR04InlineWholeFileMath(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "source")
	if err := os.Mkdir(source, 0755); err != nil { t.Fatal(err) }
	const size = 512 * 1024
	pattern := []byte("deduparc-neutron-dag-period-31!")
	root := make([]byte, size)
	for i := range root {root[i]=pattern[i%len(pattern)]}
	expected := map[string][]byte{}
	for n:=0; n<8; n++ {
		data:=bytes.Clone(root)
		for j:=0; j<2*n; j++ { at:=1024+j*17011; data[at]^=byte(11+n+j) }
		name:=fmt.Sprintf("formula-%02d.bin",n)
		if err:=os.WriteFile(filepath.Join(source,name),data,0600);err!=nil {t.Fatal(err)}
		expected[name]=data
	}
	cfg:=chunker.Config{Min:size, Avg:size, Max:size}
	r03,err:=Create(filepath.Join(dir,"r03.darc"),[]string{source},CreateOptions{Compression:CompressionNeutronDAG,Chunker:cfg})
	if err!=nil {t.Fatal(err)}
	r04,err:=Create(filepath.Join(dir,"r04.darc"),[]string{source},CreateOptions{Compression:CompressionNeutronDense,Chunker:cfg})
	if err!=nil {t.Fatal(err)}
	if r04.Manifest.Version!=6 {t.Fatalf("version %d !=6",r04.Manifest.Version)}
	if r04.Manifest.Stats.InlineMathFiles != 8 {t.Fatalf("inline files: %d",r04.Manifest.Stats.InlineMathFiles)}
	if len(r04.Manifest.MathRoots)!=1 {t.Fatalf("roots=%d, want 1",len(r04.Manifest.MathRoots))}
	if r04.Bytes>=r03.Bytes {t.Fatalf("density regression: R03=%d R04=%d",r03.Bytes,r04.Bytes)}

	r,err:=Open(filepath.Join(dir,"r04.darc"));if err!=nil {t.Fatal(err)}
	defer r.Close()
	if _,err:=r.Verify();err!=nil{t.Fatal(err)}
	out:=filepath.Join(dir,"restored")
	if err:=r.Extract(out);err!=nil {t.Fatal(err)}
	for name,want:=range expected {
		got,err:=os.ReadFile(filepath.Join(out,"source",name))
		if err!=nil {t.Fatal(err)}
		if !bytes.Equal(got,want) {t.Fatalf("mismatch: %s",name)}
	}
	old,err:=Open(filepath.Join(dir,"r03.darc"));if err!=nil{t.Fatal(err)}
	defer old.Close()
	if _,err:=old.Verify();err!=nil{t.Fatal(err)}
	t.Logf("source=%d R03=%d R04=%d inline=%d roots=%d",8*size,r03.Bytes,r04.Bytes,r04.Manifest.Stats.InlineMathFiles,len(r04.Manifest.MathRoots))
}

func TestNeutronR04PureMathWithoutPhysicalChunks(t *testing.T) {
	dir:=t.TempDir()
	data:=make([]byte,2*1024*1024)
	for i:=range data {data[i]=byte(13+i*7)}
	p:=filepath.Join(dir,"affine.bin")
	if err:=os.WriteFile(p,data,0600);err!=nil{t.Fatal(err)}
	result,err:=Create(filepath.Join(dir,"pure.darc"),[]string{p},CreateOptions{Compression:CompressionNeutronDense})
	if err!=nil{t.Fatal(err)}
	if result.Manifest.Stats.InlineMathFiles!=1 || result.Manifest.Stats.UniqueChunks!=0 {
		t.Fatalf("expected virtual-only math file: %+v",result.Manifest.Stats)
	}
	r,err:=Open(filepath.Join(dir,"pure.darc"));if err!=nil{t.Fatal(err)}
	defer r.Close()
	if len(r.chunks)!=0{t.Fatalf("unexpected physical chunks: %d",len(r.chunks))}
	if _,err:=r.Verify();err!=nil{t.Fatal(err)}
}

func TestNeutronR04InlineRejectsInvalidReferences(t *testing.T) {
	data:=bytes.Repeat([]byte("A"),128*1024)
	code,ok:=mathTry(data,100)
	if !ok || !isPureMathCode(code){t.Fatal("missing pure base")}
	r:=&Reader{Manifest:Manifest{MathRoots:[]InlineMathRoot{{Size:uint32(len(data)),Code:code}}}}
	encoded:=binary.AppendUvarint(nil,0)
	encoded=binary.AppendUvarint(encoded,0)
	good,err:=r.decodeInlineMath(Entry{Size:int64(len(data)),InlineMath:encoded})
	if err!=nil||!bytes.Equal(good,data){t.Fatalf("valid failed %v",err)}
	for _,bad:=range [][]byte{
		{1,0}, // missing root
		{0,1,0xff}, // truncated gap
		{0,1,0xff,0xff,0x7f,'x'}, // out of range
		{0,0,0}, // trailing data
	} {
		if _,err:=r.decodeInlineMath(Entry{Size:int64(len(data)),InlineMath:bad});err==nil{
			t.Fatalf("accepted invalid inline ref %x",bad)
		}
	}
	if _,err:=r.decodeInlineMath(Entry{Size:int64(len(data))+1,InlineMath:encoded});err==nil{t.Fatal("accepted mismatched size")}
}
