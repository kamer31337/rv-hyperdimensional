#ifndef RV_HDC_H
#define RV_HDC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RV_KERNEL_SINC = 0,
    RV_KERNEL_GAUSSIAN = 1
} rv_kernel_type_t;

typedef struct {
    size_t dim;
    double *data;
} rv_vector_t;

typedef struct {
    size_t dim;
    double beta;
    rv_kernel_type_t kernel_type;
    double *frequencies;
    double *phases;
} rv_hdc_config_t;

typedef struct {
    size_t dim;
    rv_vector_t sin_part;
    rv_vector_t cos_part;
    rv_vector_t full_vec;
} rv_sinusoid_vec_t;

typedef struct {
    size_t dim;
    rv_vector_t r_sin;
    rv_vector_t r_cos;
    rv_vector_t r_full;
} rv_seq_repr_b_t;

typedef struct {
    double mean;
    double variance;
    double std_dev;
    double min_val;
    double max_val;
} rv_dist_stats_t;

/* PRNG and Sampling API */
void rv_prng_seed(uint64_t seed);
double rv_random_uniform(double min_val, double max_val);
double rv_random_gaussian(double mean, double std_dev);
int rv_sample_frequencies(rv_hdc_config_t *cfg, uint64_t seed);
int rv_sample_random_phases(rv_hdc_config_t *cfg, uint64_t seed);

/* Config Management */
int rv_hdc_config_init(rv_hdc_config_t *cfg, size_t dim, double beta, rv_kernel_type_t kernel_type, uint64_t seed);
void rv_hdc_config_free(rv_hdc_config_t *cfg);

/* Vector Memory and Core Operations */
int rv_vec_alloc(rv_vector_t *vec, size_t dim);
void rv_vec_free(rv_vector_t *vec);
void rv_vec_zero(rv_vector_t *vec);
void rv_vec_copy(rv_vector_t *dst, const rv_vector_t *src);
void rv_vec_scale(rv_vector_t *vec, double scale);
void rv_vec_add(rv_vector_t *dst, const rv_vector_t *src);
void rv_vec_sub(rv_vector_t *dst, const rv_vector_t *src);
void rv_vec_hadamard(rv_vector_t *dst, const rv_vector_t *a, const rv_vector_t *b);
double rv_vec_dot(const rv_vector_t *a, const rv_vector_t *b);
double rv_vec_norm(const rv_vector_t *vec);
double rv_vec_norm_sq(const rv_vector_t *vec);
double rv_vec_cosine_sim(const rv_vector_t *a, const rv_vector_t *b);
rv_dist_stats_t rv_vec_compute_stats(const rv_vector_t *vec);

/* Variant A: Standard FPE Baseline via Hermitian IDFT */
int rv_fpe_a_compute_position(const rv_hdc_config_t *cfg, double t, rv_vector_t *out_vec);
int rv_seq_encode_variant_a(const rv_hdc_config_t *cfg, const rv_vector_t *elem_vecs, const double *positions, size_t seq_len, rv_vector_t *out_repr);

/* Variant B: Real-Valued Sinusoid RFF with Exact Shift Operator */
int rv_sinusoid_vec_alloc(rv_sinusoid_vec_t *svec, size_t dim);
void rv_sinusoid_vec_free(rv_sinusoid_vec_t *svec);
int rv_fpe_b_compute_position(const rv_hdc_config_t *cfg, double t, rv_sinusoid_vec_t *out_svec);
int rv_fpe_b_shift_vector(const rv_sinusoid_vec_t *in_pos, const rv_sinusoid_vec_t *shift_pos, rv_sinusoid_vec_t *out_shifted);

int rv_fpe_b_seq_alloc(rv_seq_repr_b_t *seq_repr, size_t dim);
void rv_fpe_b_seq_free(rv_seq_repr_b_t *seq_repr);
void rv_fpe_b_seq_zero(rv_seq_repr_b_t *seq_repr);
int rv_fpe_b_seq_encode_step(rv_seq_repr_b_t *seq_repr, const rv_vector_t *elem_vec, const rv_sinusoid_vec_t *pos_vec);
int rv_fpe_b_seq_encode(const rv_hdc_config_t *cfg, const rv_vector_t *elem_vecs, const double *positions, size_t seq_len, rv_seq_repr_b_t *out_seq);
int rv_fpe_b_seq_shift(const rv_seq_repr_b_t *in_seq, const rv_sinusoid_vec_t *shift_pos, rv_seq_repr_b_t *out_shifted_seq);

/* Variant C: Real-Valued Cosine-Only Representation */
int rv_fpe_c_compute_position(const rv_hdc_config_t *cfg, double t, rv_vector_t *out_vec);
int rv_seq_encode_variant_c(const rv_hdc_config_t *cfg, const rv_vector_t *elem_vecs, const double *positions, size_t seq_len, rv_vector_t *out_repr);

/* Evaluation and Analysis Utilities */
double rv_theoretical_kernel(rv_kernel_type_t kernel_type, double tau, double beta);
double rv_eval_kernel_sample_a(const rv_hdc_config_t *cfg, double t1, double t2);
double rv_eval_kernel_sample_b(const rv_hdc_config_t *cfg, double t1, double t2);
double rv_eval_kernel_sample_c(const rv_hdc_config_t *cfg, double t1, double t2);
double rv_find_best_shift_b(const rv_seq_repr_b_t *seq_base, const rv_seq_repr_b_t *seq_target, const rv_hdc_config_t *cfg, double shift_min, double shift_max, double shift_step, double *out_best_similarity);
int rv_generate_synthetic_peak_signal(rv_vector_t *elem_vecs, size_t seq_len, size_t dim, size_t peak_pos, double peak_height, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* RV_HDC_H */
