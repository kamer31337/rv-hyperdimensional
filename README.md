# Real-Valued Hyperdimensional Sequence Representations in C11

A high-performance C11 implementation of real-valued position encodings with **Hadamard Product Binding** and **Exact Shift Equivariance** for Hyperdimensional Computing / Vector Symbolic Architectures (HDC/VSA), based on:

> **"Real-Valued Hyperdimensional Sequence Representations with Hadamard Product Binding and Shift Equivariance"**  
> Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Amy Loutfi, Stefan Streif, Evgeny Osipov (arXiv:2608.28334, 2026).

---

## Key Features

- **Hadamard Product Compatibility**: Operates entirely in real-valued vector space ($\mathbb{R}^D$ and $\mathbb{R}^{2D}$) using element-wise multiplication ($\odot$), eliminating complex-valued arithmetic.
- **Exact Algebraic Shift Operator ($\diamond$)**: Computes temporal shifts directly on accumulated sequence representations in $O(D)$ time using trigonometric angle-addition identities:
  $$r_\delta = r \diamond \psi(\delta, \beta) \equiv \sum_{t=1}^T \tilde{f}_t \odot \psi(t + \delta, \beta)$$
- **Shift-Invariant Kernels**: Supports **Sinc** ($\mathcal{U}(-\pi, \pi)$) and **Gaussian** ($\mathcal{N}(0, \sigma^2)$) similarity kernels via Random Fourier Features (RFF).
- **Strict Norm Stability**: Guarantees constant unit norm $\|\psi(t, \beta)\|^2 = 1.0$ across all continuous time steps $t$.
- **ISO C11 Standard**: Pure C11 (`-std=c11`) with zero third-party dependencies, adhering to strict coding standards.

---

## Comparison of Implemented Variants

| Property | Variant A (Baseline FPE) | Variant B (Sinusoid RFF) | Variant C (Cosine-Only RFF) |
| :--- | :--- | :--- | :--- |
| **Vector Space** | $\mathbb{R}^D$ | $\mathbb{R}^{2D}$ ($[\psi_{\sin}^T, \psi_{\cos}^T]^T$) | $\mathbb{R}^D$ |
| **Formula** | $p(t, \beta) = \mathcal{F}^{-1}(c^{\beta t + 1})$ | $\psi(t, \beta) = \frac{1}{\sqrt{D}} [\sin(\omega \beta t), \cos(\omega \beta t)]^T$ | $\chi(t, \beta) = \sqrt{\frac{2}{D}} \cos(\omega \beta t + b)$ |
| **Binding Operation** | Real Hadamard ($\odot$) | Real Hadamard ($\odot$) | Real Hadamard ($\odot$) |
| **Norm Stability** | Constant | **Exact Constant ($1.0$)** | Expected Constant ($1.0$) |
| **Shift Equivariance** | No | **Exact Algebraic ($r \diamond \psi(\delta)$)** | No |
| **Shift Complexity** | $O(T D \log D)$ | **$O(D)$** | $O(T D)$ |

---

## Building and Running

### Using CMake

```bash
mkdir build && cd build
cmake ..
cmake --build .
./rv_hdc_demo
```

### Using Make (GCC / Clang)

```bash
make
./bin/rv_hdc_demo
```

### Using MSVC (Visual Studio Developer Command Prompt)

```cmd
cl /std:c11 /O2 /Iinclude src\*.c examples\demo.c /Fe:rv_hdc_demo.exe
rv_hdc_demo.exe
```

---

## Code Example: Variant B Sequence Encoding & Shifting

```c
#include "rv_hdc.h"
#include <stdio.h>

int main(void)
{
    size_t dim = 1000;
    double beta = 1.0;
    
    /* 1. Initialize configuration with Sinc kernel */
    rv_hdc_config_t cfg;
    rv_hdc_config_init(&cfg, dim, beta, RV_KERNEL_SINC, 42ULL);

    /* 2. Allocate sequence representation */
    rv_seq_repr_b_t seq_repr;
    rv_fpe_b_seq_alloc(&seq_repr, dim);

    /* 3. Encode sequence elements */
    rv_vector_t feature;
    rv_sinusoid_vec_t pos_vec;
    rv_vec_alloc(&feature, dim);
    rv_sinusoid_vec_alloc(&pos_vec, dim);

    for (size_t t = 0; t < 50; t++) {
        /* Populate feature vector ... */
        rv_fpe_b_compute_position(&cfg, (double)t, &pos_vec);
        rv_fpe_b_seq_encode_step(&seq_repr, &feature, &pos_vec);
    }

    /* 4. Apply exact temporal shift by delta = 10.0 in O(D) time */
    rv_sinusoid_vec_t shift_vec;
    rv_sinusoid_vec_alloc(&shift_vec, dim);
    rv_fpe_b_compute_position(&cfg, 10.0, &shift_vec);

    rv_seq_repr_b_t shifted_repr;
    rv_fpe_b_seq_alloc(&shifted_repr, dim);
    rv_fpe_b_seq_shift(&seq_repr, &shift_vec, &shifted_repr);

    printf("Original norm: %.6f, Shifted norm: %.6f\n", 
           rv_vec_norm(&seq_repr.r_full), rv_vec_norm(&shifted_repr.r_full));

    /* Clean up */
    rv_fpe_b_seq_free(&shifted_repr);
    rv_sinusoid_vec_free(&shift_vec);
    rv_sinusoid_vec_free(&pos_vec);
    rv_vec_free(&feature);
    rv_fpe_b_seq_free(&seq_repr);
    rv_hdc_config_free(&cfg);
    return 0;
}
```

---

## Documentation

See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for full mathematical proofs, kernel derivation, component value distributions, and comprehensive API documentation.

## THEORETISTS NAMES
 ____________________________________________________
| Mentioned Names might be fictional propagated over |
| network not real theoretists |
 ------------------------------