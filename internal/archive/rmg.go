package archive

import (
    "encoding/binary"
    "fmt"
)

// R05: a bounded, lossless, separable matrix predictor over Z/256Z.
// For each byte in a row-major matrix, M[row,col] = U[col] + V[row]
// (mod 256), plus exact sparse replacements. The parameter vectors U and V
// are themselves encoded using the existing mathematical bytecode if that
// reduces their size. Thus decoding has two deterministic model levels,
// not arbitrary executable code or unlimited recursion.
const (
    methodRMG byte = 5
    rmgWireVersion byte = 1
    rmgMaxRaw = 8 * 1024 * 1024
    rmgMinRaw = 4096
    rmgMaxWidth = 1024
    rmgMinRows = 16
    rmgProbeRows = 16
)
var rmgWidths = [...]int{32,64,128,256,512,1024}

// The wire format is:
// version:u8, width:uvarint,
// U:[encoding:u8 (raw=0,math=1), byte_len:uvarint, payload],
// V:[encoding:u8, byte_len:uvarint, payload],
// patches:uvarint, repeated (gap:uvarint, exact_byte:u8).
// The original byte count is already in the DARC CHNK record.
type rmgReader struct {
    b []byte
    pos int
}
func (r *rmgReader) u() (uint64,error) {
    if r.pos>=len(r.b) { return 0,ErrInvalidArchive }
    n,used:=binary.Uvarint(r.b[r.pos:])
    if used<=0 {return 0,ErrInvalidArchive}
    r.pos+=used
    return n,nil
}
func (r *rmgReader) vec(want int) ([]byte,error) {
    if want<1 || want>rmgMaxRaw || r.pos>=len(r.b) {return nil,ErrInvalidArchive}
    kind:=r.b[r.pos];r.pos++
    length,err:=r.u()
    if err!=nil || length>uint64(len(r.b)-r.pos) {return nil,ErrInvalidArchive}
    encoded:=r.b[r.pos:r.pos+int(length)]
    r.pos+=int(length)
    switch kind {
    case 0:
        if len(encoded)!=want {return nil,fmt.Errorf("rmg: raw vector length mismatch")}
        return encoded,nil
    case 1:
        if len(encoded)<3 || len(encoded)>=want {return nil,fmt.Errorf("rmg: invalid nested mathematical vector")}
        return mathDecode(encoded,uint32(want))
    default:
        return nil,fmt.Errorf("rmg: unknown vector codec %d",kind)
    }
}

func rmgDecode(code []byte,rawSize uint32) ([]byte,error) {
    if rawSize<rmgMinRaw || rawSize>rmgMaxRaw || len(code)<8 || code[0]!=rmgWireVersion {
        return nil,fmt.Errorf("rmg: invalid header or raw size")
    }
    r:=rmgReader{b:code,pos:1}
    width64,err:=r.u()
    if err!=nil || width64<32 || width64>rmgMaxWidth ||
        width64&(width64-1)!=0 || uint64(rawSize)%width64!=0 {
        return nil,fmt.Errorf("rmg: invalid matrix shape")
    }
    width:=int(width64)
    rows:=int(rawSize)/width
    if rows<rmgMinRows {return nil,fmt.Errorf("rmg: too few rows")}
    u,err:=r.vec(width);if err!=nil{return nil,err}
    v,err:=r.vec(rows);if err!=nil{return nil,err}
    if v[0]!=0 {return nil,fmt.Errorf("rmg: first row offset must be zero")}
    dst:=make([]byte,int(rawSize))
    for row:=0;row<rows;row++{
        base:=row*width
        for col:=0;col<width;col++{dst[base+col]=u[col]+v[row]}
    }
    patches,err:=r.u()
    if err!=nil || patches>uint64(rawSize) {return nil,fmt.Errorf("rmg: patch count out of range")}
    var prev uint64
    for i:=uint64(0);i<patches;i++ {
        gap,err:=r.u();if err!=nil{return nil,err}
        if prev>=uint64(rawSize) || gap>=uint64(rawSize)-prev || r.pos>=len(code) {
            return nil,fmt.Errorf("rmg: patch offset out of range")
        }
        index:=prev+gap
        dst[index]=code[r.pos];r.pos++
        prev=index+1
    }
    if r.pos!=len(code) {return nil,fmt.Errorf("rmg: trailing bytes")}
    return dst,nil
}

func rmgProbe(data []byte,width int) bool {
    rows:=len(data)/width
    if rows<rmgMinRows {return false}
    selected:=rmgProbeRows
    if selected>rows-1 { selected=rows-1 }
    // Use evenly spaced rows. No incomplete source sample can prove
    // correctness; acceptance later checks every single input byte.
    for s:=0;s<selected;s++ {
        row:=1+(s*(rows-2))/selected
        var count [256]uint16
        origin:=row*width
        most:=uint16(0)
        for j:=0;j<width;j++{
            delta:=data[origin+j]-data[j]
            count[delta]++
            if count[delta]>most { most=count[delta] }
        }
        if int(most)*8<width*7 { return false }
    }
    return true
}
func rmgPutVector(out []byte,vector []byte) []byte {
    packed,ok:=mathTryDense(vector,len(vector))
    if ok && len(packed)+2<len(vector) {
        out=append(out,1)
        out=binary.AppendUvarint(out,uint64(len(packed)))
        return append(out,packed...)
    }
    out=append(out,0)
    out=binary.AppendUvarint(out,uint64(len(vector)))
    return append(out,vector...)
}

// rmgTry returns a candidate only when its complete payload (including
// recursive vector models and every byte correction) substantially beats
// the caller's current best packed payload.
func rmgTry(data []byte,bestBytes int) ([]byte,bool) {
    if len(data)<rmgMinRaw || len(data)>rmgMaxRaw || bestBytes<32 {
        return nil,false
    }
    var best []byte
    for _,width:=range rmgWidths {
        if len(data)%width!=0 || len(data)/width<rmgMinRows || !rmgProbe(data,width) {
            continue
        }
        rows:=len(data)/width
        u:=data[:width]
        offsets:=make([]byte,rows)
        // Find the mode of (row - base row), robust to sparse edits.
        good:=true
        for row:=1;row<rows;row++ {
            start:=row*width
            var count [256]uint16
            mode,most:=byte(0),uint16(0)
            for j:=0;j<width;j++ {
                delta:=data[start+j]-u[j]
                count[delta]++
                if count[delta]>most {mode,most=delta,count[delta]}
            }
            offsets[row]=mode
            if int(most)*4<width*3 {good=false;break}
        }
        if !good {continue}

        header:=[]byte{rmgWireVersion}
        header=binary.AppendUvarint(header,uint64(width))
        header=rmgPutVector(header,u)
        header=rmgPutVector(header,offsets)
        limit:=bestBytes
        if best!=nil && len(best)<limit {limit=len(best)}
        if len(header)+1+16>=limit {continue}

        edits:=make([]byte,0,128)
        count:=uint64(0)
        prev:=0
        for i,b:=range data {
            predicted:=u[i%width]+offsets[i/width]
            if predicted==b {continue}
            edits=binary.AppendUvarint(edits,uint64(i-prev))
            edits=append(edits,b)
            count++
            prev=i+1
            if len(header)+len(edits)+10+16>=limit {break}
        }
        if len(header)+len(edits)+10+16>=limit {continue}
        code:=binary.AppendUvarint(header,count)
        code=append(code,edits...)
        if len(code)+16<limit && len(code)<len(data) {
            best=code
        }
    }
    return best,best!=nil
}
