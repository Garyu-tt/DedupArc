package archive

import "encoding/binary"

// The quadratic basis uses binomial coefficients instead of floating point:
// f(i)=a+b*i+c*i*(i-1)/2 (mod 256). This is exact integer arithmetic.
// R04-only; legacy R02/R03 writers keep using the original opcode set.
func triangularMod(i int) byte {
	if i < 2 { return 0 }
	return byte((uint64(i) * uint64(i-1) / 2) & 255)
}

// The optional quadratic path does not change the outputs of old v4/v5 modes.
func mathTryDense(data []byte, maxBytes int) ([]byte, bool) {
	best, ok := mathTry(data, maxBytes)
	limit := maxBytes
	if ok && len(best)<limit { limit=len(best) }
	quad, found := mathTryQuadratic(data, limit)
	if found { return quad, true }
	return best, ok
}

func mathTryQuadratic(data []byte, maxBytes int) ([]byte, bool) {
	if len(data)<64 || len(data)>mathMaxRaw || maxBytes<5 {
		return nil,false
	}
	sample:=len(data);if sample>mathSample {sample=mathSample}
	var hist [256]int

	// The dominant second difference survives sparse point mutations:
	// one edited sample byte affects at most three local second differences.
	for i:=2;i<sample;i++ {
		second:=data[i]-byte(2)*data[i-1]+data[i-2]
		hist[second]++
	}
	curvature,most:=byte(0),0
	for i,n:=range hist {if n>most {curvature,most=byte(i),n}}
	if most < sample/2 {return nil,false}

	hist=[256]int{}
	for i:=1;i<sample;i++ {
		first:=data[i]-data[i-1]-byte(i-1)*curvature
		hist[first]++
	}
	step,most:=byte(0),0
	for i,n:=range hist {if n>most {step,most=byte(i),n}}
	if most<sample/2 {return nil,false}

	hist=[256]int{}
	for i:=0;i<sample;i++ {
		intercept:=data[i]-byte(i)*step-triangularMod(i)*curvature
		hist[intercept]++
	}
	a,correct:=byte(0),0
	for i,n:=range hist {if n>correct {a,correct=byte(i),n}}
	if correct < sample/2 {return nil,false}
	estimated:=sample-correct
	if estimated*2*len(data)/sample > maxBytes*2 {return nil,false}

	plan:=mathPlan{op:mathQuadratic,a:a,step:step,curvature:curvature}
	prefix:=[]byte{mathQuadratic,a,step,curvature}
	patches:=make([]byte,0,128)
	count,prev:=uint64(0),0
	for i,b:=range data {
		if plan.value(i)==b {continue}
		patches=binary.AppendUvarint(patches,uint64(i-prev))
		patches=append(patches,b)
		prev=i+1
		count++
		if len(prefix)+len(patches)+10>=maxBytes {return nil,false}
	}
	out:=binary.AppendUvarint(prefix,count)
	out=append(out,patches...)
	if len(out)>=maxBytes || len(out)>=len(data) {return nil,false}
	return out,true
}
