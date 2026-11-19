package archive

import (
	"encoding/binary"
	"fmt"
)

// R02 deterministic math bytecode: bounded, lossless formula + sparse exceptions.
// A DARC chunk record already contains the uncompressed byte count.
const (
	methodMath byte = 3
	mathConstant byte = 0xA0
	mathAffine byte = 0xA1
	mathPeriodic byte = 0xA2
	mathMaxPeriod = 32
	mathMaxRaw = 32 * 1024 * 1024
	mathSample = 4096
)

type mathPlan struct {
	op byte
	a, step byte
	pattern []byte
	estimated int
}

func (p mathPlan) value(i int) byte {
	switch p.op {
	case mathConstant:
		return p.a
	case mathAffine:
		return p.a + byte(i)*p.step
	default:
		return p.pattern[i%len(p.pattern)]
	}
}

// mathTry never produces an output larger than maxBytes. It may reject patterns
// rather than spend unbounded CPU searching for an optimal expression.
func mathTry(data []byte, maxBytes int) ([]byte, bool) {
	if len(data) < 64 || len(data) > mathMaxRaw || maxBytes < 3 {
		return nil, false
	}
	n := len(data)
	sample := n
	if sample > mathSample { sample = mathSample }
	plans := make([]mathPlan, 0, 3)
	var counts [256]int
	for i:=0; i<sample; i++ { counts[data[i]]++ }
	mode, freq := byte(0), 0
	for i,c := range counts { if c>freq { mode,freq=byte(i),c } }
	plans=append(plans, mathPlan{op:mathConstant,a:mode,estimated:sample-freq})

	// Infer the dominant modulo-256 slope and intercept, tolerating sparse edits.
	counts=[256]int{}
	for i:=1; i<sample; i++ { counts[data[i]-data[i-1]]++ }
	step, maxFreq:=byte(0),0
	for i,c:=range counts { if c>maxFreq { step,maxFreq=byte(i),c } }
	counts=[256]int{}
	for i:=0; i<sample; i++ { counts[data[i]-byte(i)*step]++ }
	start, matches:=byte(0),0
	for i,c:=range counts { if c>matches { start,matches=byte(i),c } }
	plans=append(plans, mathPlan{op:mathAffine,a:start,step:step,estimated:sample-matches})

	// Find the shortest likely period using a capped sample.
	bestPeriodic:=mathPlan{estimated:sample+1}
	for period:=2; period<=mathMaxPeriod && period<=sample/4; period++ {
		var hist [mathMaxPeriod][256]uint16 // <= 4096 sample positions
		for i:=0; i<sample; i++ { hist[i%period][data[i]]++ }
		pattern:=make([]byte,period)
		correct:=0
		for slot:=0; slot<period; slot++ {
			most:=0
			for v,count:=range hist[slot] { if int(count)>most { most=int(count); pattern[slot]=byte(v) } }
			correct+=most
		}
		misses:=sample-correct
		if misses*2+period < bestPeriodic.estimated*2+len(bestPeriodic.pattern) {
			bestPeriodic=mathPlan{op:mathPeriodic,pattern:pattern,estimated:misses}
		}
	}
	if bestPeriodic.op!=0 { plans=append(plans,bestPeriodic) }

	var best []byte
	for _,p:=range plans {
		// A patch requires at least two bytes, so near-random samples are poor candidates.
		if p.estimated > sample/2 { continue }
		// A cheap heuristic only; the complete source is checked losslessly below.
		if p.estimated*2*len(data)/sample > maxBytes*2 { continue }
		prefix:=[]byte{p.op}
		switch p.op {
		case mathConstant: prefix=append(prefix,p.a)
		case mathAffine: prefix=append(prefix,p.a,p.step)
		case mathPeriodic: prefix=append(prefix,byte(len(p.pattern))); prefix=append(prefix,p.pattern...)
		}
		max:=maxBytes
		if best!=nil && len(best)<max { max=len(best) }
		patches:=make([]byte,0,128)
		count,prev:=uint64(0),0
		valid:=true
		for i,b:=range data {
			if p.value(i)==b { continue }
			patches=binary.AppendUvarint(patches,uint64(i-prev))
			patches=append(patches,b)
			prev=i+1
			count++
			if len(prefix)+len(patches)>max {
				valid=false; break
			}
		}
		if !valid {continue}
		out:=binary.AppendUvarint(prefix,count)
		out=append(out,patches...)
		if len(out)<max && len(out)<len(data) {best=out}
	}
	return best,best!=nil
}

func mathDecode(code []byte, rawSize uint32) ([]byte,error) {
	if rawSize>mathMaxRaw || len(code)<3 {return nil,fmt.Errorf("math bytecode: invalid sizes")}
	p:=mathPlan{op:code[0]}
	cursor:=1
	switch p.op {
	case mathConstant:
		if len(code)<cursor+1 {return nil,fmt.Errorf("math bytecode: truncated constant")}
		p.a=code[cursor];cursor++
	case mathAffine:
		if len(code)<cursor+2 {return nil,fmt.Errorf("math bytecode: truncated affine")}
		p.a,p.step=code[cursor],code[cursor+1];cursor+=2
	case mathPeriodic:
		if len(code)<cursor+1 {return nil,fmt.Errorf("math bytecode: truncated period")}
		period:=int(code[cursor]);cursor++
		if period<1 || period>mathMaxPeriod || len(code)<cursor+period {return nil,fmt.Errorf("math bytecode: invalid period")}
		p.pattern=code[cursor:cursor+period];cursor+=period
	default:
		return nil,fmt.Errorf("math bytecode: unknown opcode %x",p.op)
	}
	count,used:=binary.Uvarint(code[cursor:])
	if used<=0 || count>uint64(rawSize) {return nil,fmt.Errorf("math bytecode: invalid patch count")}
	cursor+=used
	dst:=make([]byte,int(rawSize))
	for i:=range dst {dst[i]=p.value(i)}
	prev:=uint64(0)
	for j:=uint64(0);j<count;j++ {
		if cursor>=len(code) {return nil,fmt.Errorf("math bytecode: truncated patch")}
		gap,n:=binary.Uvarint(code[cursor:]);if n<=0 {return nil,fmt.Errorf("math bytecode: invalid patch delta")}
		cursor+=n
		if gap>uint64(rawSize) || prev>uint64(rawSize)-gap || prev+gap>=uint64(rawSize) || cursor>=len(code) {return nil,fmt.Errorf("math bytecode: patch out of bounds")}
		idx:=prev+gap
		dst[idx]=code[cursor];cursor++
		prev=idx+1
	}
	if cursor!=len(code) {return nil,fmt.Errorf("math bytecode: trailing data")}
	return dst,nil
}
