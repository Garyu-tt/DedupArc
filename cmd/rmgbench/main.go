// rmgbench is a fixed-seed synthetic research fixture; it never reads user files.
package main

import (
 "bytes"
 "compress/zlib"
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "fmt"
 "math/rand"
 "os"
 "time"

 "github.com/Garyu-tt/deduparc/internal/rmg"
)
type result struct {
 Corpus string `json:"corpus"`
 Seed int64 `json:"seed"`
 SourceSHA256 string `json:"source_sha256"`
 OriginalBytes int `json:"original_bytes"`
 ZlibPayloadBytes int `json:"zlib_payload_bytes"`
 RMGEncodedBytes int `json:"rmg_encoded_bytes"`
 RMGMode string `json:"rmg_mode"`
 CompressMS int64 `json:"compress_ms"`
 DecodeMS int64 `json:"decode_ms"`
 BitIdentical bool `json:"bit_identical"`
}
func zlibSize(src []byte) int {
 var out bytes.Buffer
 z,_:=zlib.NewWriterLevel(&out,zlib.BestCompression)
 _,_=z.Write(src);_=z.Close()
 return out.Len()
}
func patterned(width,rows int,seed int64,corruptEvery int) []byte {
 g:=rand.New(rand.NewSource(seed));base:=make([]byte,width);_,_=g.Read(base)
 out:=make([]byte,width*rows)
 for row:=0;row<rows;row++ {
  shift:=byte(g.Intn(256))
  for col:=0;col<width;col++ {out[row*width+col]=base[col]+shift}
  if corruptEvery>0&&row%corruptEvery==0 {out[row*width+(row%width)]^=0xb7}
 }
 return out
}
func samples() map[string][]byte {
 g:=rand.New(rand.NewSource(20261119))
 random:=make([]byte,131072);_,_=g.Read(random)
 nestedBase:=patterned(16,64,113,0)
 nestedShifts:=patterned(16,16,114,0)
 nested:=make([]byte,1024*256)
 for row:=0;row<256;row++ {for col:=0;col<1024;col++ {nested[row*1024+col]=nestedBase[col]+nestedShifts[row]}}
 repeated:=bytes.Repeat([]byte("constant repeatable payload entry\n"),4096)
 configs:=bytes.Repeat([]byte("{\"name\":\"node\",\"enabled\":true,\"id\":42}\n"),3072)
 versions:=make([]byte,256*256);copy(versions[:256],random[:256])
 for r:=1;r<256;r++ {
  copy(versions[r*256:(r+1)*256],versions[:256])
  for i:=0;i<4;i++ {versions[r*256+(r*31+i*59)%256]^=byte(i+r)}
 }
 return map[string][]byte{
  "structured_rows":patterned(256,256,20261119,23),
  "nested_matrix_rules":nested,
  "similar_binary_versions":versions,
  "repetitive_text":repeated,
  "config_text":configs,
  "random_uniform":random,
 }
}
func main() {
 if len(os.Args)!=1 {fmt.Fprintln(os.Stderr,"no arguments supported: synthetic fixed-seed corpus only");os.Exit(2)}
 order:=[]string{"structured_rows","nested_matrix_rules","similar_binary_versions","repetitive_text","config_text","random_uniform"}
 encoder:=json.NewEncoder(os.Stdout)
 for _,name:=range order {
  src:=samplesCache[name]
  digest:=sha256.Sum256(src)
  begin:=time.Now();code,err:=rmg.Encode(src);encodeMS:=time.Since(begin).Milliseconds()
  if err!=nil {panic(err)}
  begin=time.Now();restored,err:=rmg.Decode(code);decodeMS:=time.Since(begin).Milliseconds()
  verified:=err==nil&&bytes.Equal(src,restored)
  item:=result{Corpus:name,Seed:20261119,SourceSHA256:hex.EncodeToString(digest[:]),OriginalBytes:len(src),ZlibPayloadBytes:zlibSize(src),RMGEncodedBytes:len(code),RMGMode:rmg.TopMode(code),CompressMS:encodeMS,DecodeMS:decodeMS,BitIdentical:verified}
  if err=encoder.Encode(item);err!=nil {panic(err)}
  if !verified {os.Exit(1)}
 }
}
var samplesCache=samples()
