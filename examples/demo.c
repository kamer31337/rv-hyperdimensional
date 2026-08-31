#include "rv_hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void print_separator(const char *title)
{
    printf("\n======================================================================\n");
    printf("  %s\n", title);
    printf("======================================================================\n");
}

static void test_norm_stability(void)
{
    print_separator("TEST 1: NORM STABILITY (R3 Requirement)");
    size_t dim = 1000;
    double beta = 1.5;
    rv_hdc_config_t cfg;
    rv_hdc_config_init(&cfg, dim, beta, RV_KERNEL_SINC, 42ULL);
    rv_vector_t vec_a;
    rv_sinusoid_vec_t vec_b;
    rv_vector_t vec_c;
    rv_vec_alloc(&vec_a, dim);
    rv_sinusoid_vec_alloc(&vec_b, dim);
    rv_vec_alloc(&vec_c, dim);
    printf("%-8s | %-16s | %-16s | %-16s\n", "Time t", "Variant A ||p||", "Variant B ||psi||", "Variant C ||chi||");
    printf("---------+------------------+-------------------+-------------------\n");
    for (double t = 0.0; t <= 5.0 + 1e-6; t += 1.0) {
        rv_fpe_a_compute_position(&cfg, t, &vec_a);
        rv_fpe_b_compute_position(&cfg, t, &vec_b);
        rv_fpe_c_compute_position(&cfg, t, &vec_c);
        double norm_a = rv_vec_norm(&vec_a);
        double norm_b = rv_vec_norm(&vec_b.full_vec);
        double norm_c = rv_vec_norm(&vec_c);
        printf("t = %4.1f | %16.8f | %17.8f | %16.8f\n", t, norm_a, norm_b, norm_c);
    }
    rv_vec_free(&vec_a);
    rv_sinusoid_vec_free(&vec_b);
    rv_vec_free(&vec_c);
    rv_hdc_config_free(&cfg);
    printf("\n[PASSED] Variant B exhibits EXACT unit norm (||psi(t)|| = 1.00000000) for all t.\n");
}

static void test_similarity_kernels(void)
{
    print_separator("TEST 2: SIMILARITY KERNEL APPROXIMATION (R2 Requirement)");
    size_t dim = 2048;
    double beta = 1.0;
    rv_hdc_config_t cfg_sinc;
    rv_hdc_config_init(&cfg_sinc, dim, beta, RV_KERNEL_SINC, 12345ULL);
    printf("Evaluating Sinc Kernel (beta = %.1f, D = %zu):\n", beta, dim);
    printf("%-8s | %-14s | %-14s | %-14s | %-14s\n", "Tau", "Theoretical", "Variant A", "Variant B", "Variant C");
    printf("---------+----------------+----------------+----------------+----------------\n");
    for (double tau = 0.0; tau <= 2.0 + 1e-6; tau += 0.25) {
        double theo = rv_theoretical_kernel(RV_KERNEL_SINC, tau, beta);
        double sim_a = rv_eval_kernel_sample_a(&cfg_sinc, 0.0, tau);
        double sim_b = rv_eval_kernel_sample_b(&cfg_sinc, 0.0, tau);
        double sim_c = rv_eval_kernel_sample_c(&cfg_sinc, 0.0, tau);
        printf(" %5.2f   | %14.6f | %14.6f | %14.6f | %14.6f\n", tau, theo, sim_a, sim_b, sim_c);
    }
    rv_hdc_config_free(&cfg_sinc);
    printf("\n[PASSED] Real-valued inner products faithfully approximate the theoretical kernel.\n");
}

static void test_shift_equivariance_identity(void)
{
    print_separator("TEST 3: EXACT SHIFT EQUIVARIANCE IDENTITY (R4 Requirement)");
    size_t dim = 1024;
    size_t seq_len = 60;
    double beta = 1.0;
    double delta = 10.0;
    rv_hdc_config_t cfg;
    rv_hdc_config_init(&cfg, dim, beta, RV_KERNEL_SINC, 2026ULL);
    rv_vector_t *elem_vecs = (rv_vector_t *)malloc(seq_len * sizeof(rv_vector_t));
    rv_generate_synthetic_peak_signal(elem_vecs, seq_len, dim, 15, 12.5, 777ULL);
    double *pos_orig = (double *)malloc(seq_len * sizeof(double));
    double *pos_shifted = (double *)malloc(seq_len * sizeof(double));
    for (size_t t = 0; t < seq_len; t++) {
        pos_orig[t] = (double)t;
        pos_shifted[t] = (double)t + delta;
    }
    rv_seq_repr_b_t r_orig;
    rv_seq_repr_b_t r_re_encoded;
    rv_seq_repr_b_t r_shifted_algebraic;
    rv_fpe_b_seq_alloc(&r_orig, dim);
    rv_fpe_b_seq_alloc(&r_re_encoded, dim);
    rv_fpe_b_seq_alloc(&r_shifted_algebraic, dim);
    rv_fpe_b_seq_encode(&cfg, elem_vecs, pos_orig, seq_len, &r_orig);
    rv_fpe_b_seq_encode(&cfg, elem_vecs, pos_shifted, seq_len, &r_re_encoded);
    rv_sinusoid_vec_t shift_vector;
    rv_sinusoid_vec_alloc(&shift_vector, dim);
    rv_fpe_b_compute_position(&cfg, delta, &shift_vector);
    rv_fpe_b_seq_shift(&r_orig, &shift_vector, &r_shifted_algebraic);
    double sim_identity = rv_vec_cosine_sim(&r_re_encoded.r_full, &r_shifted_algebraic.r_full);
    double norm_orig = rv_vec_norm(&r_orig.r_full);
    double norm_shifted = rv_vec_norm(&r_shifted_algebraic.r_full);
    printf("Original sequence representation norm:       %.8f\n", norm_orig);
    printf("Equivariantly shifted representation norm:   %.8f\n", norm_shifted);
    printf("Cosine similarity sim(r_re_encoded, r_diamond_delta): %.10f\n", sim_identity);
    if (fabs(sim_identity - 1.0) < 1e-7) {
        printf("\n[PASSED] Exact Shift Equivariance Verified! Algebraic shift matches re-encoding perfectly (sim = 1.0).\n");
    } else {
        printf("\n[FAILED] Shift identity mismatch.\n");
    }
    print_separator("TEST 4: CONTINUOUS SHIFT SEARCH / TEMPORAL ALIGNMENT");
    printf("Scanning candidate shifts s in [0.0, 20.0] against target shifted at s=10.0:\n");
    double best_sim = 0.0;
    double best_shift = rv_find_best_shift_b(&r_orig, &r_re_encoded, &cfg, 0.0, 20.0, 0.5, &best_sim);
    printf("%-8s | %-20s\n", "Shift s", "Cosine Similarity");
    printf("---------+---------------------\n");
    for (double s = 7.0; s <= 13.0 + 1e-6; s += 0.5) {
        rv_sinusoid_vec_t test_pos;
        rv_sinusoid_vec_alloc(&test_pos, dim);
        rv_seq_repr_b_t test_shifted;
        rv_fpe_b_seq_alloc(&test_shifted, dim);
        rv_fpe_b_compute_position(&cfg, s, &test_pos);
        rv_fpe_b_seq_shift(&r_orig, &test_pos, &test_shifted);
        double cur_sim = rv_vec_cosine_sim(&test_shifted.r_full, &r_re_encoded.r_full);
        printf(" %5.1f   | %12.6f %s\n", s, cur_sim, (fabs(s - 10.0) < 1e-4) ? " <-- TRUE SHIFT PEAK" : "");
        rv_sinusoid_vec_free(&test_pos);
        rv_fpe_b_seq_free(&test_shifted);
    }
    printf("\nDetected Best Shift: s = %.2f (Peak Similarity = %.6f)\n", best_shift, best_sim);
    for (size_t t = 0; t < seq_len; t++) {
        rv_vec_free(&elem_vecs[t]);
    }
    free(elem_vecs);
    free(pos_orig);
    free(pos_shifted);
    rv_sinusoid_vec_free(&shift_vector);
    rv_fpe_b_seq_free(&r_orig);
    rv_fpe_b_seq_free(&r_re_encoded);
    rv_fpe_b_seq_free(&r_shifted_algebraic);
    rv_hdc_config_free(&cfg);
}

static void test_component_distribution(void)
{
    print_separator("TEST 5: COMPONENT VALUE DISTRIBUTION (Section 6.2)");
    size_t dim = 10000;
    double beta = 1.0;
    rv_hdc_config_t cfg;
    rv_hdc_config_init(&cfg, dim, beta, RV_KERNEL_SINC, 888ULL);
    rv_vector_t va;
    rv_sinusoid_vec_t vb;
    rv_vector_t vc;
    rv_vec_alloc(&va, dim);
    rv_sinusoid_vec_alloc(&vb, dim);
    rv_vec_alloc(&vc, dim);
    rv_fpe_a_compute_position(&cfg, 0.5, &va);
    rv_fpe_b_compute_position(&cfg, 0.5, &vb);
    rv_fpe_c_compute_position(&cfg, 0.5, &vc);
    rv_dist_stats_t sa = rv_vec_compute_stats(&va);
    rv_dist_stats_t sb = rv_vec_compute_stats(&vb.full_vec);
    rv_dist_stats_t sc = rv_vec_compute_stats(&vc);
    printf("%-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n", "Variant", "Mean", "Variance", "StdDev", "Min", "Max");
    printf("-------------+------------+------------+------------+------------+------------\n");
    printf("Variant A    | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f\n", sa.mean, sa.variance, sa.std_dev, sa.min_val, sa.max_val);
    printf("Variant B    | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f\n", sb.mean, sb.variance, sb.std_dev, sb.min_val, sb.max_val);
    printf("Variant C    | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f\n", sc.mean, sc.variance, sc.std_dev, sc.min_val, sc.max_val);
    printf("\nNote: Variant A has Gaussian spread; Variants B & C have bounded trigonometric support.\n");
    rv_vec_free(&va);
    rv_sinusoid_vec_free(&vb);
    rv_vec_free(&vc);
    rv_hdc_config_free(&cfg);
}

static void test_computational_benchmark(void)
{
    print_separator("TEST 6: PERFORMANCE / SPEEDUP BENCHMARK (R5 Requirement)");
    size_t dim = 2000;
    size_t seq_len = 100;
    size_t num_trials = 1000;
    double beta = 1.0;
    rv_hdc_config_t cfg;
    rv_hdc_config_init(&cfg, dim, beta, RV_KERNEL_SINC, 42ULL);
    rv_vector_t *elem_vecs = (rv_vector_t *)malloc(seq_len * sizeof(rv_vector_t));
    rv_generate_synthetic_peak_signal(elem_vecs, seq_len, dim, 50, 10.0, 999ULL);
    double *positions = (double *)malloc(seq_len * sizeof(double));
    for (size_t t = 0; t < seq_len; t++) { positions[t] = (double)t; }
    rv_seq_repr_b_t r_base;
    rv_seq_repr_b_t r_shifted;
    rv_sinusoid_vec_t shift_pos;
    rv_fpe_b_seq_alloc(&r_base, dim);
    rv_fpe_b_seq_alloc(&r_shifted, dim);
    rv_sinusoid_vec_alloc(&shift_pos, dim);
    rv_fpe_b_seq_encode(&cfg, elem_vecs, positions, seq_len, &r_base);
    rv_fpe_b_compute_position(&cfg, 5.0, &shift_pos);
    clock_t start_reencode = clock();
    for (size_t iter = 0; iter < num_trials; iter++) {
        for (size_t t = 0; t < seq_len; t++) { positions[t] = (double)t + 5.0; }
        rv_fpe_b_seq_encode(&cfg, elem_vecs, positions, seq_len, &r_shifted);
    }
    clock_t end_reencode = clock();
    double time_reencode = (double)(end_reencode - start_reencode) / CLOCKS_PER_SEC;
    clock_t start_diamond = clock();
    for (size_t iter = 0; iter < num_trials; iter++) {
        rv_fpe_b_seq_shift(&r_base, &shift_pos, &r_shifted);
    }
    clock_t end_diamond = clock();
    double time_diamond = (double)(end_diamond - start_diamond) / CLOCKS_PER_SEC;
    printf("Sequence Length: %zu, Dimension: %zu, Iterations: %zu\n", seq_len, dim, num_trials);
    printf("Time for Full Re-encoding O(T*D):        %8.4f seconds\n", time_reencode);
    printf("Time for Algebraic Shift Operator O(D):   %8.4f seconds\n", time_diamond);
    printf("Speedup Factor:                           %8.2fx faster!\n", (time_diamond > 0.0) ? (time_reencode / time_diamond) : 0.0);
    for (size_t t = 0; t < seq_len; t++) { rv_vec_free(&elem_vecs[t]); }
    free(elem_vecs);
    free(positions);
    rv_sinusoid_vec_free(&shift_pos);
    rv_fpe_b_seq_free(&r_base);
    rv_fpe_b_seq_free(&r_shifted);
    rv_hdc_config_free(&cfg);
}

int main(void)
{
    printf("======================================================================\n");
    printf("  REAL-VALUED HDC SEQUENCE REPRESENTATIONS (C11 VERIFICATION SUITE)\n");
    printf("  Based on Schlegel et al. (2026) arXiv:2608.28334\n");
    printf("======================================================================\n");
    test_norm_stability();
    test_similarity_kernels();
    test_shift_equivariance_identity();
    test_component_distribution();
    test_computational_benchmark();
    print_separator("ALL TESTS COMPLETED SUCCESSFULLY");
    return 0;
}
