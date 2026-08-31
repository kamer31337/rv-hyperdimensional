#include "rv_hdc.h"
#include <math.h>
#include <stdlib.h>

int rv_fpe_c_compute_position(const rv_hdc_config_t *cfg, double t, rv_vector_t *out_vec)
{
    if (cfg == NULL || out_vec == NULL || out_vec->data == NULL) { return -1; }
    size_t d = cfg->dim;
    double scale = sqrt(2.0 / (double)d);
    double beta_t = cfg->beta * t;
    for (size_t i = 0; i < d; i++) {
        double angle = cfg->frequencies[i] * beta_t + cfg->phases[i];
        out_vec->data[i] = scale * cos(angle);
    }
    return 0;
}

int rv_seq_encode_variant_c(const rv_hdc_config_t *cfg, const rv_vector_t *elem_vecs, const double *positions, size_t seq_len, rv_vector_t *out_repr)
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
        rv_fpe_c_compute_position(cfg, positions[t], &pos_vec);
        rv_vec_hadamard(&bound_vec, &elem_vecs[t], &pos_vec);
        rv_vec_add(out_repr, &bound_vec);
    }
    rv_vec_free(&bound_vec);
    rv_vec_free(&pos_vec);
    return 0;
}
