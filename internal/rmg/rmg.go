// Package rmg implements a bounded, experimental lossless recursive matrix model.
// It is an optional research codec, not part of the .darc format or live runtime.
package rmg

import (
 "bytes"
 "compress/zlib"
 "encoding/binary"
 "errors"
 "fmt"
 "io"
)

const (
 MaxBytes = 2 << 20
 MaxDepth = 3
 magic = "RMG1"
 modeRaw byte = 0
 modeZlib byte = 1
 modeMatrix byte = 2
)

var ErrInvalid = errors.New("rmg: malformed or unsupported stream")

type writer struct{ bytes.Buffer }
func (w *writer) num(x uint64) { var b [10]byte; n:=binary.PutUvarint(b[:], x); w.Write(b[:n]) }

type reader struct { b []byte; off int }
func (r *reader) num() (uint64,error) {
 if r.off>=len(r.b) { return 0,ErrInvalid }
 x,n:=binary.Uvarint(r.b[r.off:]); if n<=0 { return 0,ErrInvalid }; r.off+=n;return x,nil
}
func (r *reader) part(n uint64) ([]byte,error) {
 if n>uint64(len(r.b)-r.off) { return nil,ErrInvalid }
 start:=r.off;r.off+=int(n);return r.b[start:r.off],nil
}
func (r *reader) end() bool { return r.off==len(r.b) }

// Encode chooses the smallest byte-accurate candidate including all model and
// residual overhead, with raw and zlib always competing against matrix models.
func Encode(data []byte) ([]byte,error) {
 if len(data)>MaxBytes {return nil,fmt.Errorf("rmg: input exceeds %d bytes",MaxBytes)}
 return encode(data,MaxDepth),nil
}
func encode(data []byte,depth int) []byte {
 best:=raw(data)
 if z:=zipped(data);len(z)<len(best) {best=z}
 if depth==0 || len(data)<256 {return best}
 for _,width:=range []int{16,32,64,128,256,512,1024} {
  if len(data)%width!=0 || len(data)/width<2 {continue}
  for _,step:=range []byte{1,2,4,8,16,32} {
   if candidate:=matrix(data,width,step,depth-1,len(best));len(candidate)>0&&len(candidate)<len(best) {best=candidate}
  }
 }
 return best
}
func header(mode byte,n int) *writer {
 w:=new(writer);w.WriteString(magic);w.WriteByte(mode);w.num(uint64(n));return w
}
func raw(data []byte) []byte {w:=header(modeRaw,len(data));w.Write(data);return w.Bytes()}
func zipped(data []byte) []byte {
 var compressed bytes.Buffer
 z,_:=zlib.NewWriterLevel(&compressed,zlib.BestCompression)
 _,_=z.Write(data);_=z.Close()
 w:=header(modeZlib,len(data));w.num(uint64(compressed.Len()));w.Write(compressed.Bytes());return w.Bytes()
}
func quantize(x,step byte) byte {
 // Round to nearest power-of-two multiple mod 256.
 return byte((uint16(x)+uint16(step)/2)&^uint16(step-1))
}
func matrix(data []byte,width int,step byte,depth,ceiling int) []byte {
 rows:=len(data)/width
 base:=append([]byte(nil),data[:width]...)
 shifts:=make([]byte,rows)
 for row:=1;row<rows;row++ {
  var freq [256]int
  off:=row*width
  for col:=0;col<width;col++ {freq[byte(int(data[off+col])-int(base[col]))]++}
  mode,count:=0,-1
  for j,n:=range freq {if n>count {mode,count=j,n}}
  shifts[row]=quantize(byte(mode),step)
 }
 var residual writer
 // Exact deviations: delta-coded index + literal, never lossy.
 mismatches:=0;prev:=-1
 for row:=0;row<rows;row++ {for col:=0;col<width;col++ {
  index:=row*width+col
  if predicted:=base[col]+shifts[row];predicted!=data[index] {
   residual.num(uint64(index-prev-1));residual.WriteByte(data[index]);prev=index;mismatches++
  }
 }}
 // Reject expensive candidates before spending effort recursively.
 if residual.Len()+mismatches+width+rows >=ceiling {return nil}
 baseCode:=encode(base,depth);shiftCode:=encode(shifts,depth)
 w:=header(modeMatrix,len(data));w.num(uint64(width));w.WriteByte(step)
 w.num(uint64(len(baseCode)));w.num(uint64(len(shiftCode)))
 w.num(uint64(mismatches));w.Write(baseCode);w.Write(shiftCode);w.Write(residual.Bytes())
 if w.Len()>=ceiling {return nil};return w.Bytes()
}

// Decode enforces per-node output bounds, recursion depth and complete input
// consumption before returning reconstructed bytes.
func Decode(data []byte) ([]byte,error) {return decode(data,MaxDepth)}
func decode(data []byte,depth int) ([]byte,error) {
 if len(data)<6||string(data[:4])!=magic{return nil,ErrInvalid}
 mode:=data[4];r:=reader{b:data,off:5}
 length,err:=r.num();if err!=nil||length>MaxBytes {return nil,ErrInvalid}
 n:=int(length)
 switch mode {
 case modeRaw:
  payload,e:=r.part(length);if e!=nil||!r.end(){return nil,ErrInvalid}
  return append([]byte(nil),payload...),nil
 case modeZlib:
  size,e:=r.num();if e!=nil{return nil,ErrInvalid}
  payload,e:=r.part(size);if e!=nil||!r.end(){return nil,ErrInvalid}
  z,e:=zlib.NewReader(bytes.NewReader(payload));if e!=nil{return nil,ErrInvalid}
  result,e:=io.ReadAll(io.LimitReader(z,int64(n)+1));closeErr:=z.Close()
  if e!=nil||closeErr!=nil||len(result)!=n {return nil,ErrInvalid}
  return result,nil
 case modeMatrix:
  if depth==0||n<256 {return nil,ErrInvalid}
  width64,e:=r.num();if e!=nil||width64<16||width64>1024||length%width64!=0 {return nil,ErrInvalid}
  width:=int(width64);rows:=n/width
  if rows<2 {return nil,ErrInvalid}
  stepPart,e:=r.part(1);if e!=nil{return nil,ErrInvalid}
  step:=stepPart[0];if step==0||step>32||step&(step-1)!=0 {return nil,ErrInvalid}
  baseLen,e:=r.num();if e!=nil{return nil,ErrInvalid}
  shiftLen,e:=r.num();if e!=nil{return nil,ErrInvalid}
  mismatches,e:=r.num();if e!=nil||mismatches>length{return nil,ErrInvalid}
  baseCode,e:=r.part(baseLen);if e!=nil{return nil,ErrInvalid}
  shiftCode,e:=r.part(shiftLen);if e!=nil{return nil,ErrInvalid}
  base,e:=decode(baseCode,depth-1);if e!=nil||len(base)!=width{return nil,ErrInvalid}
  shifts,e:=decode(shiftCode,depth-1);if e!=nil||len(shifts)!=rows{return nil,ErrInvalid}
  out:=make([]byte,n)
  for row:=0;row<rows;row++ {for col:=0;col<width;col++ {out[row*width+col]=base[col]+shifts[row]}}
  prev:=-1
  for i:=uint64(0);i<mismatches;i++ {
   gap,e:=r.num();if e!=nil||gap>=uint64(n-prev){return nil,ErrInvalid}
   index:=prev+1+int(gap);if index>=n {return nil,ErrInvalid}
   literal,e:=r.part(1);if e!=nil {return nil,ErrInvalid}
   out[index]=literal[0];prev=index
  }
  if !r.end(){return nil,ErrInvalid}
  return out,nil
 default: return nil,ErrInvalid
 }
}

// TopMode describes the winner, not proof that recursion is present.
func TopMode(data []byte) string {
 if len(data)<5||string(data[:4])!=magic{return "invalid"}
 switch data[4] {case modeRaw:return "raw";case modeZlib:return "zlib";case modeMatrix:return "matrix"}
 return "invalid"
}
