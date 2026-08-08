package updating

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"testing"
)

func TestVersionCompare(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{{"0.5.1", "0.5.0", 1}, {"0.5.0", "0.5.0", 0}, {"0.4.9", "0.5.0", -1}, {"1.0.0", "0.9.9", 1}}
	for _, c := range cases {
		if got := compareVersions(c.a, c.b); got != c.want {
			t.Fatalf("%s vs %s = %d", c.a, c.b, got)
		}
	}
}

func TestSignManifestAcceptsSeed(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	seed := priv.Seed()
	key := base64.StdEncoding.EncodeToString(seed)
	data := []byte(`{"version":"test"}`)
	if _, err := SignManifest(data, key); err != nil {
		t.Fatal(err)
	}
}
