package rmg

import (
 "bytes"
 "math/rand"
 "testing"
)

func structured(width,rows int) []byte {
 rng:=rand.New(rand.NewSource(20261119))
 base:=make([]byte,width);rng.Read(base)
 data:=make([]byte,width*rows)
 for r:=0;r<rows;r++ {
  shift:=byte(rng.Intn(256))
  for c:=0;c<width;c++ {data[r*width+c]=base[c]+shift}
  if r>0&&r%23==0 {data[r*width+r%width]^=0x4d}
 }
 return data
}
func TestMatrixRoundtripAndSavings(t *testing.T) {
 data:=structured(256,256)
 code,e:=Encode(data);if e!=nil {t.Fatal(e)}
 got,e:=Decode(code);if e!=nil||!bytes.Equal(got,data){t.Fatalf("roundtrip: %v",e)}
 if TopMode(code)!="matrix" {t.Fatalf("want matrix, got %s",TopMode(code))}
 if len(code)>=len(zipped(data)) {t.Fatalf("model lost to zlib: %d vs %d",len(code),len(zipped(data)))}
 t.Logf("input=%d RMG=%d zlib-only=%d",len(data),len(code),len(zipped(data)))
}
func TestRandomAndFallback(t *testing.T) {
 rng:=rand.New(rand.NewSource(42))
 for _,n:=range []int{0,1,31,255,256,256*3,1<<16} {
  data:=make([]byte,n);rng.Read(data)
  code,e:=Encode(data);if e!=nil{t.Fatal(e)}
  got,e:=Decode(code);if e!=nil||!bytes.Equal(got,data){t.Fatalf("n=%d e=%v",n,e)}
  if len(code)>len(raw(data)){t.Fatalf("fallback grows: %d > %d",len(code),len(raw(data))}
 }
}
func TestRecursiveRulePacking(t *testing.T) {
 data:=structured(128,256)
 code,e:=Encode(data);if e!=nil {t.Fatal(e)}
 out,e:=Decode(code);if e!=nil||!bytes.Equal(out,data){t.Fatal("recursive rule roundtrip",e)}
}
func TestCorruptionRejected(t *testing.T) {
 data:=structured(256,64)
 code,e:=Encode(data);if e!=nil {t.Fatal(e)}
 cases:=[][]byte{nil,[]byte("RMG1"),append([]byte(nil),code[:len(code)-1]...),append([]byte(nil),code...)}
 cases[3][4]=99
 for i,b:=range cases {if _,e:=Decode(b);e==nil{t.Errorf("case %d accepted",i)}}
 for _,n:=range []int{MaxBytes+1} {if _,e:=Encode(make([]byte,n));e==nil {t.Error("oversize accepted")}}
}
func TestExactResidual(t *testing.T) {
 data:=structured(512,128)
 for i:=0;i<len(data);i+=211 {data[i]^=byte(i*7)}
 code,_:=Encode(data);got,e:=Decode(code)
 if e!=nil||!bytes.Equal(got,data){t.Fatal("residual mismatch",e)}
}

func countMatrixNodes(code []byte) int {
 if len(code)<5||string(code[:4])!=magic{return 0}
 if code[4]!=modeMatrix{return 0}
 r:=reader{b:code,off:5}
 _,_ = r.num()
 _,_ = r.num()
 _,_ = r.part(1)
 baseSize,_:=r.num()
 shiftSize,_:=r.num()
 _,_=r.num()
 a,_:=r.part(baseSize);b,_:=r.part(shiftSize)
 return 1+countMatrixNodes(a)+countMatrixNodes(b)
}
func TestNestedRecursiveRulesReallyUsed(t *testing.T) {
 base:=structured(16,64)
 shifts:=structured(16,16)
 data:=make([]byte,1024*256)
 for row:=0;row<256;row++ {for col:=0;col<1024;col++ {
  data[row*1024+col]=base[col]+shifts[row]
 }}
 code,e:=Encode(data);if e!=nil {t.Fatal(e)}
 got,e:=Decode(code);if e!=nil||!bytes.Equal(got,data){t.Fatal("roundtrip failed",e)}
 nodes:=countMatrixNodes(code)
 t.Logf("recursive corpus input=%d encoded=%d matrix_nodes=%d",len(data),len(code),nodes)
 if nodes<2 {t.Fatalf("need genuine recursively modeled rules, observed %d node(s)",nodes)}
}
func TestUntrustedInputNeverPanics(t *testing.T) {
 rand:=rand.New(rand.NewSource(115))
 for i:=0;i<3000;i++ {
  n:=rand.Intn(128);input:=make([]byte,n);rand.Read(input)
  if n>=4&&i%5==0 {copy(input,[]byte(magic))}
  _,_=Decode(input)
 }
}
func TestQuantizedRulesWithExactResidual(t *testing.T) {
 data:=structured(256,64)
 code:=matrix(data,256,16,MaxDepth-1,MaxBytes+1024)
 if len(code)==0 {t.Fatal("matrix candidate unexpectedly rejected")}
 restored,e:=Decode(code)
 if e!=nil||!bytes.Equal(data,restored) {t.Fatalf("lossless quantization residual failed: %v",e)}
}
func TestTamperedLengthAndTrailingBytes(t *testing.T) {
 input:=structured(128,32)
 code,e:=Encode(input);if e!=nil {t.Fatal(e)}
 if _,e:=Decode(append(append([]byte(nil),code...),0));e==nil {t.Fatal("accepted trailing bytes")}
 encoded:=raw(input)
 bad:=append([]byte(nil),encoded...)
 for i:=5;i<10;i++ {bad[i]=0xff}
 if _,e:=Decode(bad);e==nil {t.Fatal("accepted malicious length")}
}
