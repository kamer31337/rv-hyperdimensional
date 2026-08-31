# Real-Valued Hyperdimensional Sequence Representations with Hadamard Product Binding and Shift Equivariance

## 1. Executive Summary

In Hyperdimensional Computing / Vector Symbolic Architectures (HDC/VSA), encoding continuous sequence order and temporal proximity is traditionally achieved via **Fractional Power Encoding (FPE)**. However, standard FPE formulations in Holographic Reduced Representations (HRR) and Frequency HRR (FHRR) rely on circular convolution $\circledast$ or complex-domain component-wise multiplication, which requires $O(D \log D)$ Fast Fourier Transforms (FFT) and is incompatible with efficient real-valued vector architectures using the **Hadamard product** (component-wise multiplication $\odot$).

This library implements the mathematical framework and algorithms described by Schlegel et al. (2026), providing three real-valued position encoding variants that operate directly in $\mathbb{R}^D$ and $\mathbb{R}^{2D}$ with Hadamard product binding:
1. **Variant A (Inverse Fourier Baseline)**: Generates real vectors from Hermitian-symmetric complex phasors with an exponent offset ($\beta t + 1$). Compatible with Hadamard binding, but does not admit an algebraic shift operator ($O(D \log D)$ generation).
2. **Variant B (Sinusoid Random Fourier Features)**: Maps sampled angular frequencies $\omega$ to paired sine and cosine features in $\mathbb{R}^{2D}$. Supports an **exact algebraic shift operator $\diamond$** based on angle-addition trigonometric identities, enabling sequence transformations in purely real $O(D)$ time with constant unit norm $\|\psi(t, \beta)\|^2 = 1$.
3. **Variant C (Cosine-Only RFF)**: Uses a compact $D$-dimensional real representation with randomized phase shifts $\chi(t, \beta) = \sqrt{\frac{2}{D}}\cos(\omega\beta t + b)$. Preserves kernel similarity and unit norm in expectation $\mathbb{E}[\|\chi(t, \beta)\|^2] = 1$.

---

## 2. Mathematical Foundations

### 2.1 Role-Filler Sequence Encoding in HDC/VSA

Given a temporal sequence of feature vectors $\{f_t\}_{t=1}^T \subset \mathbb{R}^D$ at positions $t = 1, \dots, T$, the hyperdimensional sequence representation $r$ is constructed via multiplicative binding with explicit position vectors $p_t$ followed by superposition:

$$r = \sum_{t=1}^T f_t \odot p_t$$

where $\odot$ denotes component-wise (Hadamard) multiplication:

$$(a \odot b)_i = a_i \cdot b_i, \quad i = 1, \dots, D$$

### 2.2 Shift-Invariant Similarity Kernels

An explicit position vector generator $p(t, \beta)$ preserves temporal locality if its normalized inner product approximates a shift-invariant kernel function $k(t_1 - t_2)$:

$$\kappa(t_1, t_2) = \frac{1}{D} \langle p(t_1, \beta), p(t_2, \beta) \rangle \approx k(t_1 - t_2)$$

The shape of the kernel is controlled by the angular frequency distribution $\mathcal{P}$:
- **Uniform Distribution $\mathcal{U}(-\pi, \pi)$**: Yields a **Sinc kernel**:
  $$k(\tau) = \frac{\sin(\pi \beta \tau)}{\pi \beta \tau}$$
- **Gaussian Distribution $\mathcal{N}(0, \sigma^2)$**: Yields a **Gaussian / RBF kernel**:
  $$k(\tau) = \exp\left(-\frac{\sigma^2 \beta^2 \tau^2}{2}\right)$$
- **Triangular / Cauchy Distribution**: Yields compact-support or heavy-tailed kernels.

The scaling parameter $\beta > 0$ controls the bandwidth/decay rate over temporal distance $\tau = t_1 - t_2$.

---

## 3. Positional Encoding Variants

### 3.1 Variant A: Real-Valued FPE via Inverse Fourier Transform

In standard FPE, a base phasor $c = e^{i\omega}$ is raised to power $\beta t$. To prevent degeneracy at $\beta = 0$ where $c(t, 0) = [1, \dots, 1]$ becomes a single delta spike $[1, 0, \dots, 0]$ under IFFT (which would zero out all other Hadamard components), an exponent offset of $+1$ is introduced:

$$c(t, \beta) = \left[ 1, \, e^{i\omega_2(\beta t + 1)}, \, \dots, \, e^{i\omega_{D/2}(\beta t + 1)}, \, 1, \, e^{-i\omega_{D/2}(\beta t + 1)}, \, \dots, \, e^{-i\omega_2(\beta t + 1)} \right]$$

The real-valued position vector is obtained via the Inverse Discrete Fourier Transform (IDFT):

$$p(t, \beta) = \mathcal{F}^{-1}(c(t, \beta)) \in \mathbb{R}^D$$

- **Norm Stability**: Exact norm $\|p(t, \beta)\|^2 = D$ (or $1$ if normalized by $1/\sqrt{D}$).
- **Shift Equivariance**: **Not preserved** under Hadamard product ($p(t + \delta, \beta) \neq p(t, \beta) \odot p(\delta, \beta)$).
- **Component Distribution**: Zero-centered Gaussian-like distribution due to the sum of $D/2$ random cosine harmonics.

---

### 3.2 Variant B: Real-Valued Sinusoid Representation (RFF-Based)

Given $D$ sampled angular frequencies $\omega = [\omega_1, \dots, \omega_D]^T \sim \mathcal{P}^D$, we define the sine and cosine sub-vectors:

$$\psi_{\sin}(t, \beta) = \sqrt{\frac{1}{D}} \begin{bmatrix} \sin(\omega_1 \beta t) \\ \vdots \\ \sin(\omega_D \beta t) \end{bmatrix} \in \mathbb{R}^D, \qquad \psi_{\cos}(t, \beta) = \sqrt{\frac{1}{D}} \begin{bmatrix} \cos(\omega_1 \beta t) \\ \vdots \\ \cos(\omega_D \beta t) \end{bmatrix} \in \mathbb{R}^D$$

The composite position vector is formed by concatenation:

$$\psi(t, \beta) = \begin{bmatrix} \psi_{\sin}(t, \beta) \\ \psi_{\cos}(t, \beta) \end{bmatrix} \in \mathbb{R}^{2D}$$

#### Exact Unit Norm:
$$\|\psi(t, \beta)\|^2 = \|\psi_{\sin}(t, \beta)\|^2 + \|\psi_{\cos}(t, \beta)\|^2 = \frac{1}{D} \sum_{j=1}^D \left(\sin^2(\omega_j \beta t) + \cos^2(\omega_j \beta t)\right) = \frac{1}{D} \sum_{j=1}^D 1 = 1$$

#### Algebraic Shift Operator $\diamond$:
Using trigonometric angle-addition identities:
$$\sin(\omega\beta(t + \delta)) = \sin(\omega\beta t)\cos(\omega\beta\delta) + \cos(\omega\beta t)\sin(\omega\beta\delta)$$
$$\cos(\omega\beta(t + \delta)) = \cos(\omega\beta t)\cos(\omega\beta\delta) - \sin(\omega\beta t)\sin(\omega\beta\delta)$$

We define the component-wise shift operator $\diamond$ between $\psi(t, \beta)$ and $\psi(\delta, \beta)$:
$$\psi_{\sin}(t + \delta, \beta) = \psi_{\sin}(t, \beta) \odot \psi_{\cos}(\delta, \beta) + \psi_{\cos}(t, \beta) \odot \psi_{\sin}(\delta, \beta)$$
$$\psi_{\cos}(t + \delta, \beta) = \psi_{\cos}(t, \beta) \odot \psi_{\cos}(\delta, \beta) - \psi_{\sin}(t, \beta) \odot \psi_{\sin}(\delta, \beta)$$
$$\psi(t + \delta, \beta) = \psi(t, \beta) \diamond \psi(\delta, \beta)$$

#### Shift-Equivariant Sequence Representation:
To encode a sequence $\{f_t\}_{t=1}^T$ where $f_t \in \mathbb{R}^D$, we duplicate $f_t$ into $\tilde{f}_t = [f_t^T, f_t^T]^T \in \mathbb{R}^{2D}$:

$$r = \sum_{t=1}^T \tilde{f}_t \odot \psi(t, \beta) = \begin{bmatrix} r_{\sin} \\ r_{\cos} \end{bmatrix} = \begin{bmatrix} \sum_{t=1}^T f_t \odot \psi_{\sin}(t, \beta) \\ \sum_{t=1}^T f_t \odot \psi_{\cos}(t, \beta) \end{bmatrix} \in \mathbb{R}^{2D}$$

Applying a temporal shift $\delta$ directly to the accumulated sequence representation $r$:

$$r_\delta = r \diamond \psi(\delta, \beta) = \begin{bmatrix} r_{\sin} \odot \psi_{\cos}(\delta, \beta) + r_{\cos} \odot \psi_{\sin}(\delta, \beta) \\ r_{\cos} \odot \psi_{\cos}(\delta, \beta) - r_{\sin} \odot \psi_{\sin}(\delta, \beta) \end{bmatrix}$$

**Mathematical Identity**: $r_\delta \equiv \sum_{t=1}^T \tilde{f}_t \odot \psi(t + \delta, \beta)$ holds exactly, preserving $\|r_\delta\|^2 = \|r\|^2$.

---

### 3.3 Variant C: Real-Valued Cosine-Only Representation

Variant C simplifies the representation by using only cosine terms with random uniform phase offsets $b_j \sim \mathcal{U}(-\pi, \pi)$:

$$\chi(t, \beta) = \sqrt{\frac{2}{D}} \begin{bmatrix} \cos(\omega_1 \beta t + b_1) \\ \vdots \\ \cos(\omega_D \beta t + b_D) \end{bmatrix} \in \mathbb{R}^D$$

- **Norm**:
  $$\|\chi(t, \beta)\|^2 = \frac{2}{D}\sum_{j=1}^D \cos^2(\omega_j \beta t + b_j) = 1 + \frac{1}{D}\sum_{j=1}^D \cos(2\omega_j \beta t + 2b_j)$$
  $$\mathbb{E}_b\left[\|\chi(t, \beta)\|^2\right] = 1$$
- **Shift Equivariance**: **Not supported** algebraically because sine components are omitted.
- **Dimensionality**: $D$ dimensions (compact).

---

## 4. Comparison Summary

| Feature | Variant A | Variant B | Variant C |
| :--- | :--- | :--- | :--- |
| **Binding Operation** | Real Hadamard ($\odot$) | Real Hadamard ($\odot$) | Real Hadamard ($\odot$) |
| **Vector Space** | $\mathbb{R}^D$ | $\mathbb{R}^{2D}$ | $\mathbb{R}^D$ |
| **Positional Norm** | Constant $\sqrt{D}$ | Constant $1.0$ | Expected $1.0$ |
| **Exact Shift Equivariance** | No | **Yes ($r_\delta = r \diamond \psi(\delta)$)** | No |
| **Encoding Complexity** | $O(D \log D)$ | **$O(D)$** | **$O(D)$** |
| **Shift Complexity** | $O(T D \log D)$ | **$O(D)$** | $O(T D)$ |
| **Component Distribution** | Gaussian | Arcsine | Arcsine |

---

## 5. C11 Library Architecture

The library is written in pure ISO C11 (`-std=c11`) with clean modular boundaries:

```
include/
  rv_hdc.h              # Unified header: data structures and public APIs
src/
  rv_hdc_core.c         # Memory management, PRNG, vector math, similarity
  rv_hdc_fpe_a.c        # Variant A: Standard FPE baseline with IDFT
  rv_hdc_fpe_b.c        # Variant B: Sinusoid RFF with exact shift operator
  rv_hdc_fpe_c.c        # Variant C: Cosine-only RFF
  rv_hdc_sequence.c     # Sequence encoder and shift evaluation pipelines
examples/
  demo.c                # Full validation suite and performance benchmarks
```

### 5.1 Key Data Structures

```c
typedef enum {
    RV_KERNEL_SINC = 0,
    RV_KERNEL_GAUSSIAN = 1
} rv_kernel_type_t;

typedef struct {
    size_t dim;
    double beta;
    rv_kernel_type_t kernel_type;
    double *frequencies;
    double *phases;
} rv_hdc_config_t;

typedef struct {
    size_t dim;
    double *data;
} rv_vector_t;

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
```

### 5.2 Public API Overview

- **Configuration & Frequencies**:
  `rv_hdc_config_init`, `rv_hdc_config_free`, `rv_sample_frequencies`.
- **Vector Operations**:
  `rv_vec_alloc`, `rv_vec_free`, `rv_vec_hadamard`, `rv_vec_add`, `rv_vec_dot`, `rv_vec_norm`, `rv_vec_cosine_sim`.
- **Variant A**:
  `rv_fpe_a_compute_position`.
- **Variant B**:
  `rv_fpe_b_compute_position`, `rv_fpe_b_shift_vector`, `rv_fpe_b_seq_init`, `rv_fpe_b_seq_encode_step`, `rv_fpe_b_seq_shift`.
- **Variant C**:
  `rv_fpe_c_compute_position`.
- **Sequence Utilities**:
  `rv_seq_encode_variant_a`, `rv_seq_encode_variant_c`, `rv_eval_similarity_profile`.
