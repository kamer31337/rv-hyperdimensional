#include "rv_hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t g_prng_state[2] = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};

static uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t xor_shift128plus_next(void)
{
    uint64_t s0 = g_prng_state[0];
    uint64_t s1 = g_prng_state[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    g_prng_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    g_prng_state[1] = rotl(s1, 37);
    return result;
}

void rv_prng_seed(uint64_t seed)
{
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    g_prng_state[0] = z ^ (z >> 31);
    z = g_prng_state[0] + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    g_prng_state[1] = z ^ (z >> 31);
}

double rv_random_uniform(double min_val, double max_val)
{
    uint64_t r = xor_shift128plus_next();
    double unit = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return min_val + unit * (max_val - min_val);
}

double rv_random_gaussian(double mean, double std_dev)
{
    double u1 = rv_random_uniform(1e-12, 1.0);
    double u2 = rv_random_uniform(0.0, 1.0);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z0 * std_dev;
}

int rv_sample_frequencies(rv_hdc_config_t *cfg, uint64_t seed)
{
    if (cfg == NULL || cfg->frequencies == NULL) { return -1; }
    rv_prng_seed(seed);
    for (size_t i = 0; i < cfg->dim; i++) {
        if (cfg->kernel_type == RV_KERNEL_SINC) {
            cfg->frequencies[i] = rv_random_uniform(-M_PI, M_PI);
        } else if (cfg->kernel_type == RV_KERNEL_GAUSSIAN) {
            cfg->frequencies[i] = rv_random_gaussian(0.0, 1.0);
        } else {
            cfg->frequencies[i] = rv_random_uniform(-M_PI, M_PI);
        }
    }
    return 0;
}

int rv_sample_random_phases(rv_hdc_config_t *cfg, uint64_t seed)
{
    if (cfg == NULL || cfg->phases == NULL) { return -1; }
    rv_prng_seed(seed);
    for (size_t i = 0; i < cfg->dim; i++) {
        cfg->phases[i] = rv_random_uniform(-M_PI, M_PI);
    }
    return 0;
}

int rv_hdc_config_init(rv_hdc_config_t *cfg, size_t dim, double beta, rv_kernel_type_t kernel_type, uint64_t seed)
{
    if (cfg == NULL || dim == 0) { return -1; }
    cfg->dim = dim;
    cfg->beta = beta;
    cfg->kernel_type = kernel_type;
    cfg->frequencies = (double *)malloc(dim * sizeof(double));
    cfg->phases = (double *)malloc(dim * sizeof(double));
    if (cfg->frequencies == NULL || cfg->phases == NULL) {
        free(cfg->frequencies);
        free(cfg->phases);
        cfg->frequencies = NULL;
        cfg->phases = NULL;
        return -1;
    }
    rv_sample_frequencies(cfg, seed);
    rv_sample_random_phases(cfg, seed + 99991ULL);
    return 0;
}

void rv_hdc_config_free(rv_hdc_config_t *cfg)
{
    if (cfg == NULL) { return; }
    if (cfg->frequencies != NULL) {
        free(cfg->frequencies);
        cfg->frequencies = NULL;
    }
    if (cfg->phases != NULL) {
        free(cfg->phases);
        cfg->phases = NULL;
    }
    cfg->dim = 0;
}

int rv_vec_alloc(rv_vector_t *vec, size_t dim)
{
    if (vec == NULL || dim == 0) { return -1; }
    vec->dim = dim;
    vec->data = (double *)calloc(dim, sizeof(double));
    if (vec->data == NULL) {
        vec->dim = 0;
        return -1;
    }
    return 0;
}

void rv_vec_free(rv_vector_t *vec)
{
    if (vec == NULL) { return; }
    if (vec->data != NULL) {
        free(vec->data);
        vec->data = NULL;
    }
    vec->dim = 0;
}

void rv_vec_zero(rv_vector_t *vec)
{
    if (vec == NULL || vec->data == NULL) { return; }
    memset(vec->data, 0, vec->dim * sizeof(double));
}

void rv_vec_copy(rv_vector_t *dst, const rv_vector_t *src)
{
    if (dst == NULL || src == NULL || dst->data == NULL || src->data == NULL) { return; }
    size_t copy_len = (dst->dim < src->dim) ? dst->dim : src->dim;
    memcpy(dst->data, src->data, copy_len * sizeof(double));
}

void rv_vec_scale(rv_vector_t *vec, double scale)
{
    if (vec == NULL || vec->data == NULL) { return; }
    for (size_t i = 0; i < vec->dim; i++) {
        vec->data[i] *= scale;
    }
}

void rv_vec_add(rv_vector_t *dst, const rv_vector_t *src)
{
    if (dst == NULL || src == NULL || dst->data == NULL || src->data == NULL) { return; }
    size_t len = (dst->dim < src->dim) ? dst->dim : src->dim;
    for (size_t i = 0; i < len; i++) {
        dst->data[i] += src->data[i];
    }
}

void rv_vec_sub(rv_vector_t *dst, const rv_vector_t *src)
{
    if (dst == NULL || src == NULL || dst->data == NULL || src->data == NULL) { return; }
    size_t len = (dst->dim < src->dim) ? dst->dim : src->dim;
    for (size_t i = 0; i < len; i++) {
        dst->data[i] -= src->data[i];
    }
}

void rv_vec_hadamard(rv_vector_t *dst, const rv_vector_t *a, const rv_vector_t *b)
{
    if (dst == NULL || a == NULL || b == NULL) { return; }
    size_t len = dst->dim;
    if (a->dim < len) { len = a->dim; }
    if (b->dim < len) { len = b->dim; }
    for (size_t i = 0; i < len; i++) {
        dst->data[i] = a->data[i] * b->data[i];
    }
}

double rv_vec_dot(const rv_vector_t *a, const rv_vector_t *b)
{
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) { return 0.0; }
    size_t len = (a->dim < b->dim) ? a->dim : b->dim;
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

double rv_vec_norm_sq(const rv_vector_t *vec)
{
    if (vec == NULL || vec->data == NULL) { return 0.0; }
    double sum = 0.0;
    for (size_t i = 0; i < vec->dim; i++) {
        sum += vec->data[i] * vec->data[i];
    }
    return sum;
}

double rv_vec_norm(const rv_vector_t *vec)
{
    return sqrt(rv_vec_norm_sq(vec));
}

double rv_vec_cosine_sim(const rv_vector_t *a, const rv_vector_t *b)
{
    double dot = rv_vec_dot(a, b);
    double norm_a = rv_vec_norm(a);
    double norm_b = rv_vec_norm(b);
    if (norm_a <= 1e-15 || norm_b <= 1e-15) { return 0.0; }
    return dot / (norm_a * norm_b);
}

rv_dist_stats_t rv_vec_compute_stats(const rv_vector_t *vec)
{
    rv_dist_stats_t stats = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (vec == NULL || vec->data == NULL || vec->dim == 0) { return stats; }
    double sum = 0.0;
    double min_val = vec->data[0];
    double max_val = vec->data[0];
    for (size_t i = 0; i < vec->dim; i++) {
        double v = vec->data[i];
        sum += v;
        if (v < min_val) { min_val = v; }
        if (v > max_val) { max_val = v; }
    }
    double mean = sum / (double)vec->dim;
    double var_sum = 0.0;
    for (size_t i = 0; i < vec->dim; i++) {
        double diff = vec->data[i] - mean;
        var_sum += diff * diff;
    }
    double variance = var_sum / (double)vec->dim;
    stats.mean = mean;
    stats.variance = variance;
    stats.std_dev = sqrt(variance);
    stats.min_val = min_val;
    stats.max_val = max_val;
    return stats;
}

double rv_theoretical_kernel(rv_kernel_type_t kernel_type, double tau, double beta)
{
    if (kernel_type == RV_KERNEL_SINC) {
        double val = M_PI * beta * tau;
        if (fabs(val) < 1e-12) { return 1.0; }
        return sin(val) / val;
    } else if (kernel_type == RV_KERNEL_GAUSSIAN) {
        return exp(-0.5 * beta * beta * tau * tau);
    }
    return 0.0;
}
