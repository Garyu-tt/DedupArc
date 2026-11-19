package archive

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestNeutronR04QuadraticBytecode(t *testing.T) {
	const size=128*1024
	data:=make([]byte,size)
	for i:=range data {data[i]=byte(13)+byte(i)*byte(17)+triangularMod(i)*byte(5)}
	code,ok:=mathTryDense(data,1024)
	if !ok || code[0]!=mathQuadratic || len(code)!=5 {
		t.Fatalf("quadratic did not fit into five bytes: %x",code)
	}
	decoded,err:=mathDecode(code,uint32(size))
	if err!=nil || !bytes.Equal(decoded,data){t.Fatalf("exact decode failed: %v",err)}
	if old,ok:=mathTry(data,1024);ok && old[0]==mathQuadratic {
		t.Fatalf("R02 writer unexpectedly generated v6 opcode: %x",old)
	}
	rng:=rand.New(rand.NewSource(20261119))
	for i:=0;i<80;i++{
		at:=rng.Intn(size)
		data[at]^=byte(i+1)
	}
	code,ok=mathTryDense(data,4096)
	if !ok || code[0]!=mathQuadratic {t.Fatalf("patched quadratic missing: %x",code)}
	decoded,err=mathDecode(code,uint32(size))
	if err!=nil || !bytes.Equal(decoded,data){t.Fatalf("patched quadratic mismatch: %v",err)}
	for _,bad:=range [][]byte{{mathQuadratic,1,2},{mathQuadratic,1,2,3,1},{mathQuadratic,1,2,3,0,0}}{
		if _,err:=mathDecode(bad,uint32(size));err==nil {t.Fatalf("accepted malformed code: %x",bad)}
	}
}

func TestNeutronR04QuadraticWholeArchive(t *testing.T) {
	root:=t.TempDir()
	src:=filepath.Join(root,"quadratic.bin")
	data:=make([]byte,2*1024*1024)
	for i:=range data{data[i]=byte(91)+byte(i)*byte(7)+triangularMod(i)*byte(11)}
	if err:=os.WriteFile(src,data,0600);err!=nil{t.Fatal(err)}
	old,err:=Create(filepath.Join(root,"r03.darc"),[]string{src},CreateOptions{Compression:CompressionNeutronDAG})
	if err!=nil{t.Fatal(err)}
	newer,err:=Create(filepath.Join(root,"r04.darc"),[]string{src},CreateOptions{Compression:CompressionNeutronDense})
	if err!=nil{t.Fatal(err)}
	if newer.Manifest.Version!=6 || newer.Manifest.Stats.InlineMathFiles!=1 {t.Fatalf("invalid v6 stats %+v",newer.Manifest.Stats)}
	if newer.Bytes>=old.Bytes {t.Fatalf("no density gain: R03=%d R04=%d",old.Bytes,newer.Bytes)}
	r,err:=Open(filepath.Join(root,"r04.darc"));if err!=nil {t.Fatal(err)}
	defer r.Close()
	if _,err:=r.Verify();err!=nil {t.Fatal(err)}
	out:=filepath.Join(root,"restored")
	if err:=r.Extract(out);err!=nil{t.Fatal(err)}
	got,err:=os.ReadFile(filepath.Join(out,"quadratic.bin"))
	if err!=nil || !bytes.Equal(got,data){t.Fatalf("quadratic roundtrip error: %v",err)}
	t.Logf("quadratic %d source -> R03 %d vs R04 %d",len(data),old.Bytes,newer.Bytes)
}
