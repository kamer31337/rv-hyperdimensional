#include "rv_hdc.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double rv_eval_kernel_sample_a(const rv_hdc_config_t *cfg, double t1, double t2)
{
    if (cfg == NULL) { return 0.0; }
    rv_vector_t v1;
    rv_vector_t v2;
    if (rv_vec_alloc(&v1, cfg->dim) != 0) { return 0.0; }
    if (rv_vec_alloc(&v2, cfg->dim) != 0) {
        rv_vec_free(&v1);
        return 0.0;
    }
    rv_fpe_a_compute_position(cfg, t1, &v1);
    rv_fpe_a_compute_position(cfg, t2, &v2);
    double sim = rv_vec_cosine_sim(&v1, &v2);
    rv_vec_free(&v1);
    rv_vec_free(&v2);
    return sim;
}

double rv_eval_kernel_sample_b(const rv_hdc_config_t *cfg, double t1, double t2)
{
    if (cfg == NULL) { return 0.0; }
    rv_sinusoid_vec_t s1;
    rv_sinusoid_vec_t s2;
    if (rv_sinusoid_vec_alloc(&s1, cfg->dim) != 0) { return 0.0; }
    if (rv_sinusoid_vec_alloc(&s2, cfg->dim) != 0) {
        rv_sinusoid_vec_free(&s1);
        return 0.0;
    }
    rv_fpe_b_compute_position(cfg, t1, &s1);
    rv_fpe_b_compute_position(cfg, t2, &s2);
    double sim = rv_vec_dot(&s1.full_vec, &s2.full_vec);
    rv_sinusoid_vec_free(&s1);
    rv_sinusoid_vec_free(&s2);
    return sim;
}

double rv_eval_kernel_sample_c(const rv_hdc_config_t *cfg, double t1, double t2)
{
    if (cfg == NULL) { return 0.0; }
    rv_vector_t v1;
    rv_vector_t v2;
    if (rv_vec_alloc(&v1, cfg->dim) != 0) { return 0.0; }
    if (rv_vec_alloc(&v2, cfg->dim) != 0) {
        rv_vec_free(&v1);
        return 0.0;
    }
    rv_fpe_c_compute_position(cfg, t1, &v1);
    rv_fpe_c_compute_position(cfg, t2, &v2);
    double sim = rv_vec_dot(&v1, &v2);
    rv_vec_free(&v1);
    rv_vec_free(&v2);
    return sim;
}

double rv_find_best_shift_b(const rv_seq_repr_b_t *seq_base, const rv_seq_repr_b_t *seq_target, const rv_hdc_config_t *cfg, double shift_min, double shift_max, double shift_step, double *out_best_similarity)
{
    if (seq_base == NULL || seq_target == NULL || cfg == NULL || shift_step <= 0.0) { return 0.0; }
    rv_seq_repr_b_t shifted_base;
    if (rv_fpe_b_seq_alloc(&shifted_base, cfg->dim) != 0) { return 0.0; }
    rv_sinusoid_vec_t shift_pos;
    if (rv_sinusoid_vec_alloc(&shift_pos, cfg->dim) != 0) {
        rv_fpe_b_seq_free(&shifted_base);
        return 0.0;
    }
    double best_shift = shift_min;
    double best_sim = -2.0;
    for (double delta = shift_min; delta <= shift_max + 1e-9; delta += shift_step) {
        rv_fpe_b_compute_position(cfg, delta, &shift_pos);
        rv_fpe_b_seq_shift(seq_base, &shift_pos, &shifted_base);
        double sim = rv_vec_cosine_sim(&shifted_base.r_full, &seq_target->r_full);
        if (sim > best_sim) {
            best_sim = sim;
            best_shift = delta;
        }
    }
    if (out_best_similarity != NULL) {
        *out_best_similarity = best_sim;
    }
    rv_sinusoid_vec_free(&shift_pos);
    rv_fpe_b_seq_free(&shifted_base);
    return best_shift;
}

int rv_generate_synthetic_peak_signal(rv_vector_t *elem_vecs, size_t seq_len, size_t dim, size_t peak_pos, double peak_height, uint64_t seed)
{
    if (elem_vecs == NULL || seq_len == 0 || dim == 0) { return -1; }
    rv_prng_seed(seed);
    for (size_t t = 0; t < seq_len; t++) {
        if (rv_vec_alloc(&elem_vecs[t], dim) != 0) {
            for (size_t k = 0; k < t; k++) { rv_vec_free(&elem_vecs[k]); }
            return -1;
        }
        for (size_t d = 0; d < dim; d++) {
            double noise = rv_random_gaussian(0.0, 0.5);
            double base_sig = sin(2.0 * 3.141592653589793 * (double)t / 15.0);
            double val = base_sig + noise;
            if (t == peak_pos) {
                val += peak_height;
            }
            elem_vecs[t].data[d] = val;
        }
    }
    return 0;
}
