/* NeuroCrunch ZipBurger R05: fused CPU GEMV for R04 MXFP4 scale-bitpack.
 *
 * Stateless read-only kernel. Input must already pass Python CarrierR04.load()
 * SHA-256 verification. Re-check frame dimensions and bounds anyway.
 * No full dense matrix, no file access, no GPU, no heap allocation.
 *
 * Research only; identical IEEE floating-point accumulation order with
 * ggml is NOT guaranteed.
 */
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define ZBP4_HEADER 48
#define ZBP4_MAX_BLOCKS 4096

static uint16_t u16le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static unsigned scale_code(const uint8_t *data, size_t len,
                           size_t block, unsigned bits) {
    if (bits == 0) return 0;
    const size_t bitpos = block * bits;
    const size_t off = bitpos / 8;
    const unsigned shift = (unsigned)(bitpos % 8);
    unsigned code;
    if (off >= len) return 256;
    code = ((unsigned)data[off]) >> shift;
    if (shift + bits > 8) {
        if (off + 1 >= len) return 256;
        code |= ((unsigned)data[off + 1]) << (8 - shift);
    }
    return code & ((1u << bits) - 1u);
}

/* 0 means success, negative means malformed frame or invalid input.
 * Caller owns x/y; x_count and y_count are element counts, not bytes.
 */
int zipburger_r04_gemv(const uint8_t *frame, size_t size,
                       const double *x, size_t x_count,
                       double *y, size_t y_count) {
    static const double e2m1_half_units[16] = {
        0., 1., 2., 3., 4., 6., 8., 12.,
        0., -1., -2., -3., -4., -6., -8., -12.
    };
    if (frame == NULL || x == NULL || y == NULL || size < ZBP4_HEADER)
        return -1;
    if (memcmp(frame, "ZBP4", 4) != 0) return -2;

    const size_t rows = u16le(frame + 4);
    const size_t groups = u16le(frame + 6);
    const unsigned mode = frame[8];
    const unsigned bits = frame[9];
    const size_t meta_len = u16le(frame + 10);
    const size_t ix_len = u32le(frame + 12);
    if (rows == 0 || groups == 0 || rows > 2048 ||
        groups > 256 || rows * groups > ZBP4_MAX_BLOCKS)
        return -3;
    const size_t n = rows * groups;
    if (x_count != groups * 32 || y_count < rows || bits > 8)
        return -4;
    if (mode > 2) return -5;
    if ((mode == 0 && (bits != 0 || meta_len != 0 || ix_len != 0)) ||
        (mode == 1 && meta_len != 1) ||
        (mode == 2 && (meta_len == 0 || meta_len > 256)))
        return -6;
    if (mode != 0 && ix_len != (n * bits + 7) / 8)
        return -7;
    if (mode == 2) {
        unsigned required_bits = 0;
        size_t possibilities = 1;
        while (possibilities < meta_len) {
            ++required_bits;
            possibilities <<= 1;
        }
        if (bits != required_bits) return -8;
    }
    const size_t data_len = n * (mode == 0 ? 17 : 16);
    if (meta_len > size - ZBP4_HEADER) return -9;
    if (ix_len > size - ZBP4_HEADER - meta_len) return -10;
    if (data_len != size - ZBP4_HEADER - meta_len - ix_len)
        return -11;
    const uint8_t *meta = frame + ZBP4_HEADER;
    const uint8_t *index = meta + meta_len;
    const uint8_t *data = index + ix_len;

    for (size_t i = 0; i < x_count; ++i) {
        if (!isfinite(x[i])) return -12;
    }
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.;
        for (size_t g = 0; g < groups; ++g) {
            const size_t pos = r * groups + g;
            uint8_t e8m0;
            const uint8_t *packed;
            if (mode == 0) {
                e8m0 = data[pos * 17];
                packed = data + pos * 17 + 1;
            } else {
                const unsigned code = scale_code(index, ix_len, pos, bits);
                if (code > 255) return -13;
                if (mode == 1) {
                    if (code > (unsigned)(255 - meta[0])) return -14;
                    e8m0 = (uint8_t)(meta[0] + code);
                } else {
                    if (code >= meta_len) return -15;
                    e8m0 = meta[code];
                }
                packed = data + pos * 16;
            }
            double dot = 0.;
            const size_t in_start = g * 32;
            for (size_t k = 0; k < 16; ++k) {
                const uint8_t q = packed[k];
                dot += e2m1_half_units[q & 15] * x[in_start + k * 2];
                dot += e2m1_half_units[q >> 4] * x[in_start + k * 2 + 1];
            }
            sum += dot * ldexp(0.5, (int)e8m0 - 127);
        }
        y[r] = sum;
    }
    return 0;
}
