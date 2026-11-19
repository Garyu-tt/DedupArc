package archive

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCompactInlineTableRoundTrip(t *testing.T) {
	entries:=make([]Entry,0,20)
	for i:=0;i<20;i++ {
		sha:=bytes.Repeat([]byte{byte(i+1)},32)
		_ = sha
		entries=append(entries,Entry{
			Path: strings.Repeat("root/",3)+"f"+string(rune('A'+i))+".bin",
			Type:"file",Size:512*1024,Mode:0600,ModifiedUnix:1700000000,
			SHA256:strings.Repeat("a",64),InlineMath:[]byte{0,byte(i%2),1,byte(i)},
		})
	}
	m:=Manifest{Format:"deduparc",Version:6,Entries:entries,
		MathRoots:[]InlineMathRoot{{Size:512*1024,Code:[]byte{mathConstant,5,0}}}}
	raw,err:=marshalManifestV6(&m)
	if err!=nil{t.Fatal(err)}
	var decoded Manifest
	if err:=json.Unmarshal(raw,&decoded);err!=nil{t.Fatal(err)}
	if err:=expandInlineTable(&decoded);err!=nil{t.Fatal(err)}
	if len(decoded.Entries)!=len(entries) {t.Fatalf("restored %d entries, expected %d",len(decoded.Entries),len(entries))}
	for i:=range decoded.Entries {
		if decoded.Entries[i].Path!=entries[i].Path ||
			!bytes.Equal(decoded.Entries[i].InlineMath,entries[i].InlineMath) ||
			decoded.Entries[i].SHA256!=entries[i].SHA256{
			t.Fatalf("roundtrip mismatch at %d",i)
		}
	}
}

func TestCompactInlineTableCorruption(t *testing.T){
	bad:=[][]byte{
		{0x80},                       // invalid varint
		{0x01,0x03,'a'},               // truncated path
		{0x01,0x01,'f',0xff},          // invalid size
		{0x00,0x01},                  // trailing data
		{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
	}
	for _,data:=range bad {
		m:=Manifest{Version:6,InlineTable:data}
		if err:=expandInlineTable(&m);err==nil {t.Fatalf("accepted invalid inline table %x",data)}
	}
	m:=Manifest{Version:5,InlineTable:[]byte{0}}
	if err:=expandInlineTable(&m);err==nil{t.Fatal("accepted compact table in v5")}
}
