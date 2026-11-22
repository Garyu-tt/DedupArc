/* Research-only R08 3-bit E2M1 packed CPU GEMV.
 * Verify SHA-256 with palette3_r08.load() first. This C kernel checks
 * frame dimensions and shape, but deliberately has no crypto library.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#define HEADER_BYTES 48u
#define BLOCK_BYTES 13u
static const int8_t values[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static uint16_t u16le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
int zipburger_r08_gemv(const uint8_t *frame, size_t size, const double *x,
                       size_t nx, double *y, size_t ny) {
    if (!frame || !x || !y || size < HEADER_BYTES) return -1;
    if (frame[0] != 'Z' || frame[1] != 'B' || frame[2] != '3' || frame[3] != '8') return -2;
    const uint16_t rows = u16le(frame + 4), groups = u16le(frame + 6);
    const size_t total = (size_t)rows * (size_t)groups;
    if (!rows || rows > 2048 || !groups || groups > 256 || total > 4096) return -3;
    if (ny != rows || nx != (size_t)groups * 32u || size != HEADER_BYTES + BLOCK_BYTES * total) return -4;
    const uint8_t *palette = frame + 8;
    for (size_t i = 0; i < 8; ++i) {
        if (palette[i] > 15) return -5;
        for (size_t j = 0; j < i; ++j) if (palette[j] == palette[i]) return -5;
    }
    for (size_t row = 0; row < rows; ++row) {
        double out = 0.0;
        for (size_t group = 0; group < groups; ++group) {
            const uint8_t *block = frame + HEADER_BYTES + (row * groups + group) * BLOCK_BYTES;
            double dot = 0.0;
            for (size_t j = 0; j < 32; ++j) {
                const size_t bit = j * 3u, offset = bit >> 3, shift = bit & 7u;
                const uint8_t *packed = block + 1;
                uint16_t index = (uint16_t)packed[offset] >> shift;
                if (shift > 5) index |= (uint16_t)((uint16_t)packed[offset + 1] << (8u - shift));
                index &= 7u;
                dot += (double)values[palette[index]] * x[group * 32u + j];
            }
            out += dot * ldexp(0.5, (int)block[0] - 127);
        }
        y[row] = out;
    }
    return 0;
}
