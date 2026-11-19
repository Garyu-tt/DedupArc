package archive

import (
    "bytes"
    "math/rand"
    "os"
    "path/filepath"
    "testing"

    "github.com/Garyu-tt/deduparc/internal/chunker"
)

func rmgSyntheticMatrix(rows,width,changes int) []byte {
    rng:=rand.New(rand.NewSource(20261119))
    base:=make([]byte,width)
    _,_ = rng.Read(base)
    matrix:=make([]byte,rows*width)
    for row:=0;row<rows;row++{
        offset:=byte(row*11)+triangularMod(row)*5
        for col:=0;col<width;col++{
            matrix[row*width+col]=base[col]+offset
        }
    }
    // Distribute sparse independent edits away from the anchor row.
    for n:=0;n<changes;n++{
        position:=width+rng.Intn(len(matrix)-width)
        matrix[position]^=byte(1+rng.Intn(255))
    }
    return matrix
}

func TestRMGCodecRoundTrip(t *testing.T) {
    data:=rmgSyntheticMatrix(4096,128,90)
    code,ok:=rmgTry(data,len(data))
    if !ok {t.Fatal("matrix structure not discovered")}
    if len(code)>=1024 {t.Fatalf("encoded matrix too large: %d",len(code))}
    out,err:=rmgDecode(code,uint32(len(data)))
    if err!=nil {t.Fatal(err)}
    if !bytes.Equal(data,out){t.Fatal("matrix roundtrip mismatch")}
    t.Logf("RMG synthetic: %d bytes -> %d byte payload",len(data),len(code))
}

func TestRMGRejectsRandomInput(t *testing.T) {
    rng:=rand.New(rand.NewSource(20261118))
    noise:=make([]byte,512*1024)
    _,_ = rng.Read(noise)
    if code,ok:=rmgTry(noise,len(noise));ok {
        t.Fatalf("accepted incompressible random matrix: %d",len(code))
    }
}

func TestRMGMalformedStreams(t *testing.T) {
    data:=rmgSyntheticMatrix(1024,128,10)
    good,ok:=rmgTry(data,len(data))
    if !ok {t.Fatal("no baseline code")}
    if _,err:=rmgDecode(good,uint32(len(data)));err!=nil {t.Fatal(err)}
    bad:=[][]byte{
        nil,
        {0xff,0},
        good[:3],
        append(bytes.Clone(good),0),
        append([]byte{rmgWireVersion,0x80},good[2:]...),
    }
    for _,code:=range bad {
        if _,err:=rmgDecode(code,uint32(len(data)));err==nil{
            t.Fatalf("accepted malformed stream %x",code[:min(len(code),20)])
        }
    }
    if _,err:=rmgDecode(good,uint32(len(data)-1));err==nil {t.Fatal("accepted mismatched matrix length")}
}

func TestNeutronRMGArchiveRoundTrip(t *testing.T) {
    dir:=t.TempDir()
    source:=filepath.Join(dir,"matrix.bin")
    data:=rmgSyntheticMatrix(4096,128,60)
    if err:=os.WriteFile(source,data,0600);err!=nil {t.Fatal(err)}
    cfg:=chunker.Config{Min:len(data),Avg:len(data),Max:len(data)}
    previous,err:=Create(filepath.Join(dir,"r04.darc"),[]string{source},
        CreateOptions{Compression:CompressionNeutronDense,Chunker:cfg})
    if err!=nil {t.Fatal(err)}
    current,err:=Create(filepath.Join(dir,"r05.darc"),[]string{source},
        CreateOptions{Compression:CompressionNeutronRMG,Chunker:cfg})
    if err!=nil {t.Fatal(err)}
    if current.Manifest.Version!=7 || current.Manifest.Stats.RMGChunks<1 {
        t.Fatalf("RMG not selected: %+v",current.Manifest.Stats)
    }
    if current.Bytes>=previous.Bytes {
        t.Fatalf("RMG did not improve full archive: old=%d new=%d",previous.Bytes,current.Bytes)
    }
    r,err:=Open(filepath.Join(dir,"r05.darc"))
    if err!=nil {t.Fatal(err)}
    defer r.Close()
    if _,err:=r.Verify();err!=nil {t.Fatal(err)}
    out:=filepath.Join(dir,"restored")
    if err:=r.Extract(out);err!=nil {t.Fatal(err)}
    recovered,err:=os.ReadFile(filepath.Join(out,"matrix.bin"))
    if err!=nil {t.Fatal(err)}
    if !bytes.Equal(recovered,data){t.Fatal("extracted matrix mismatch")}
    t.Logf("matrix %d bytes: R04=%d B, R05=%d B",len(data),previous.Bytes,current.Bytes)
}
