#include "rv_hdc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int rv_sinusoid_vec_alloc(rv_sinusoid_vec_t *svec, size_t dim)
{
    if (svec == NULL || dim == 0) { return -1; }
    svec->dim = dim;
    if (rv_vec_alloc(&svec->full_vec, 2 * dim) != 0) { return -1; }
    svec->sin_part.dim = dim;
    svec->sin_part.data = svec->full_vec.data;
    svec->cos_part.dim = dim;
    svec->cos_part.data = svec->full_vec.data + dim;
    return 0;
}

void rv_sinusoid_vec_free(rv_sinusoid_vec_t *svec)
{
    if (svec == NULL) { return; }
    rv_vec_free(&svec->full_vec);
    svec->sin_part.data = NULL;
    svec->sin_part.dim = 0;
    svec->cos_part.data = NULL;
    svec->cos_part.dim = 0;
    svec->dim = 0;
}

int rv_fpe_b_compute_position(const rv_hdc_config_t *cfg, double t, rv_sinusoid_vec_t *out_svec)
{
    if (cfg == NULL || out_svec == NULL || out_svec->full_vec.data == NULL) { return -1; }
    size_t d = cfg->dim;
    double scale = 1.0 / sqrt((double)d);
    double beta_t = cfg->beta * t;
    for (size_t i = 0; i < d; i++) {
        double angle = cfg->frequencies[i] * beta_t;
        out_svec->sin_part.data[i] = scale * sin(angle);
        out_svec->cos_part.data[i] = scale * cos(angle);
    }
    return 0;
}

int rv_fpe_b_shift_vector(const rv_sinusoid_vec_t *in_pos, const rv_sinusoid_vec_t *shift_pos, rv_sinusoid_vec_t *out_shifted)
{
    if (in_pos == NULL || shift_pos == NULL || out_shifted == NULL) { return -1; }
    size_t d = in_pos->dim;
    if (shift_pos->dim != d || out_shifted->dim != d) { return -1; }
    double sqrt_d = sqrt((double)d);
    for (size_t i = 0; i < d; i++) {
        double s_t = in_pos->sin_part.data[i];
        double c_t = in_pos->cos_part.data[i];
        double s_d = shift_pos->sin_part.data[i];
        double c_d = shift_pos->cos_part.data[i];
        out_shifted->sin_part.data[i] = sqrt_d * (s_t * c_d + c_t * s_d);
        out_shifted->cos_part.data[i] = sqrt_d * (c_t * c_d - s_t * s_d);
    }
    return 0;
}

int rv_fpe_b_seq_alloc(rv_seq_repr_b_t *seq_repr, size_t dim)
{
    if (seq_repr == NULL || dim == 0) { return -1; }
    seq_repr->dim = dim;
    if (rv_vec_alloc(&seq_repr->r_full, 2 * dim) != 0) { return -1; }
    seq_repr->r_sin.dim = dim;
    seq_repr->r_sin.data = seq_repr->r_full.data;
    seq_repr->r_cos.dim = dim;
    seq_repr->r_cos.data = seq_repr->r_full.data + dim;
    return 0;
}

void rv_fpe_b_seq_free(rv_seq_repr_b_t *seq_repr)
{
    if (seq_repr == NULL) { return; }
    rv_vec_free(&seq_repr->r_full);
    seq_repr->r_sin.data = NULL;
    seq_repr->r_sin.dim = 0;
    seq_repr->r_cos.data = NULL;
    seq_repr->r_cos.dim = 0;
    seq_repr->dim = 0;
}

void rv_fpe_b_seq_zero(rv_seq_repr_b_t *seq_repr)
{
    if (seq_repr == NULL) { return; }
    rv_vec_zero(&seq_repr->r_full);
}

int rv_fpe_b_seq_encode_step(rv_seq_repr_b_t *seq_repr, const rv_vector_t *elem_vec, const rv_sinusoid_vec_t *pos_vec)
{
    if (seq_repr == NULL || elem_vec == NULL || pos_vec == NULL) { return -1; }
    size_t d = seq_repr->dim;
    if (elem_vec->dim != d || pos_vec->dim != d) { return -1; }
    for (size_t i = 0; i < d; i++) {
        double f_val = elem_vec->data[i];
        seq_repr->r_sin.data[i] += f_val * pos_vec->sin_part.data[i];
        seq_repr->r_cos.data[i] += f_val * pos_vec->cos_part.data[i];
    }
    return 0;
}

int rv_fpe_b_seq_encode(const rv_hdc_config_t *cfg, const rv_vector_t *elem_vecs, const double *positions, size_t seq_len, rv_seq_repr_b_t *out_seq)
{
    if (cfg == NULL || elem_vecs == NULL || positions == NULL || out_seq == NULL) { return -1; }
    rv_fpe_b_seq_zero(out_seq);
    rv_sinusoid_vec_t pos_vec;
    if (rv_sinusoid_vec_alloc(&pos_vec, cfg->dim) != 0) { return -1; }
    for (size_t t = 0; t < seq_len; t++) {
        rv_fpe_b_compute_position(cfg, positions[t], &pos_vec);
        rv_fpe_b_seq_encode_step(out_seq, &elem_vecs[t], &pos_vec);
    }
    rv_sinusoid_vec_free(&pos_vec);
    return 0;
}

int rv_fpe_b_seq_shift(const rv_seq_repr_b_t *in_seq, const rv_sinusoid_vec_t *shift_pos, rv_seq_repr_b_t *out_shifted_seq)
{
    if (in_seq == NULL || shift_pos == NULL || out_shifted_seq == NULL) { return -1; }
    size_t d = in_seq->dim;
    if (shift_pos->dim != d || out_shifted_seq->dim != d) { return -1; }
    double sqrt_d = sqrt((double)d);
    for (size_t i = 0; i < d; i++) {
        double a = in_seq->r_sin.data[i];
        double b = in_seq->r_cos.data[i];
        double s_d = shift_pos->sin_part.data[i];
        double c_d = shift_pos->cos_part.data[i];
        out_shifted_seq->r_sin.data[i] = sqrt_d * (a * c_d + b * s_d);
        out_shifted_seq->r_cos.data[i] = sqrt_d * (b * c_d - a * s_d);
    }
    return 0;
}
