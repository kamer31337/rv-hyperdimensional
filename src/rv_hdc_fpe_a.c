#include "rv_hdc.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int rv_fpe_a_compute_position(const rv_hdc_config_t *cfg, double t, rv_vector_t *out_vec)
{
    if (cfg == NULL || out_vec == NULL || out_vec->data == NULL) { return -1; }
    size_t d = cfg->dim;
    if (d < 2 || d % 2 != 0) { return -1; }
    size_t half_d = d / 2;
    double norm_factor = 1.0 / sqrt((double)d);
    double beta_t_offset = cfg->beta * t + 1.0;
    for (size_t n = 0; n < d; n++) {
        double sign_n = ((n % 2) == 0) ? 1.0 : -1.0;
        double sum_terms = 1.0 + sign_n;
        for (size_t k = 1; k < half_d; k++) {
            double omega_k = cfg->frequencies[k];
            double angle = omega_k * beta_t_offset + (2.0 * M_PI * (double)k * (double)n) / (double)d;
            sum_terms += 2.0 * cos(angle);
        }
        out_vec->data[n] = norm_factor * sum_terms;
    }
    return 0;
}

int rv_seq_encode_variant_a(const rv_hdc_config_t *cfg, const rv_vector_t *elem_vecs, const double *positions, size_t seq_len, rv_vector_t *out_repr)
{
    if (cfg == NULL || elem_vecs == NULL || positions == NULL || out_repr == NULL) { return -1; }
    rv_vec_zero(out_repr);
    rv_vector_t pos_vec;
    if (rv_vec_alloc(&pos_vec, cfg->dim) != 0) { return -1; }
    rv_vector_t bound_vec;
    if (rv_vec_alloc(&bound_vec, cfg->dim) != 0) {
        rv_vec_free(&pos_vec);
        return -1;
    }
    for (size_t t = 0; t < seq_len; t++) {
        rv_fpe_a_compute_position(cfg, positions[t], &pos_vec);
        rv_vec_hadamard(&bound_vec, &elem_vecs[t], &pos_vec);
        rv_vec_add(out_repr, &bound_vec);
    }
    rv_vec_free(&bound_vec);
    rv_vec_free(&pos_vec);
    return 0;
}
