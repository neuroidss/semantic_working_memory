
# 🧠 NeuroCanvas: Continuous SVD Phase Manifold & Zero-Lag Latent Walk Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_11.8+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-FreeEEG16_alpha2-green.svg)]()
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**NeuroCanvas** is an open-source, closed-loop Brain-Computer Interface (BCI) designed for real-time (10–12 FPS) continuous exploration of generative latent spaces. By rejecting conventional discrete motor-imagery (18–36 Hz) and trial-averaged event-related paradigms (P300/SSVEP), NeuroCanvas directly harnesses the biophysics of **Working Memory Theta-Gamma Phase Precession**. 

The system maps non-invasive scalp phase dynamics into the 768-dimensional latent manifold of multimodal foundational models (CLIP / Stable Diffusion LCM), enabling real-time, fluid cognitive navigation without text prompting or style constraints.

---

## 🔬 Theoretical Foundations & Open Neuroscience Problems

```
┌─────────────────────────────────────────────────────────────┐
│ 16-Channel Ultra-Dense Scalp Array (FreeEEG16-alpha2 @ 250Hz)│
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     [θ-Band: 6.0 ± 1.5 Hz]         [32 γ-Bands: 30..85 Hz]
               │                               │
               ▼                               │
     [Kuramoto Synchronization]                │
      Global Clock Φ_θ(t)                      │
               │                               │
               └───────────────┬───────────────┘
                               ▼
            [von Mises Theta-Gamma Multiplexing]
               32 Gaussian Phase-Locked Slices
                               │
                               ▼
            [Anchor-Referenced iPLV Extraction]
             120 Scalp Pairs vs S_0 (Past Anchor)
                               │
                               ▼
            [SVD Projection onto 49,408 Vocab Basis]
               120 Pairs ──► 768-D Continuous Drift
                               │
                               ▼
             [Linear Time-to-Sequence Unrolling]
            32 Frequency Slots ──► 77 Cross-Attention Tokens
                               │
                               ▼
           [Stable Diffusion LCM + TAESD @ 10-12 FPS]
            Zero Prompt Text Bias + Chromatic Surgery
```

### 1. Resolving the "Averaging Fallacy" (Working Memory 2.0)
* **The Scientific Gap:** Classic models posited that working memory is maintained via persistent single-neuron spiking. Modern multi-electrode findings ([Lundqvist et al., 2016, 2018](#references)) demonstrate that on single trials, delay-period activity consists of sparse, coordinated gamma-band bursts rather than sustained firing.
* **Our Solution:** NeuroCanvas bypasses trial-averaging entirely. It computes single-trial instantaneous complex phase fields across 32 dense gamma sub-bands in real time, tracking the continuous informational trajectory underlying sparse spiking.

### 2. Scalp-Level Detection of Theta-Gamma Phase Precession
* **The Scientific Gap:** The Lisman–Idiart model ([Lisman & Idiart, 1995](#references); [Heusser et al., 2016](#references)) establishes that working memory multiplexes memory items across high-frequency gamma cycles nested within a low-frequency theta oscillation (4–8 Hz). Early subcycles represent past context, while late subcycles represent look-ahead / prospective trajectories (*Vicarious Trial and Error, VTE*). This has historically been recorded almost exclusively via invasive intracranial electrophysiology.
* **Our Solution:** Utilizing an ultra-dense 26 mm circular active electrode array, NeuroCanvas measures sub-centimeter tangential phase gradients. Gating 32 gamma slices via the global Kuramoto theta phase reveals non-invasive prospective trajectories directly from the scalp.

### 3. Elimination of Zero-Lag Volume Conduction
* **The Scientific Gap:** Non-invasive scalp potentials are severely distorted by volume conduction through the cerebrospinal fluid and skull, generating spurious zero-lag synchrony.
* **Our Solution:** Following [Nolte et al. (2004)](#references) and [Bruña et al. (2018)](#references), the pipeline computes the **Imaginary Phase-Locking Value ($i\text{PLV}$)** referenced to the cycle onset anchor ($S_0$), mathematically discarding all zero-lag conduction and isolating true non-instantaneous phase interactions across 120 electrode pairs.

### 4. Bio-Resonant Closed-Loop Neuromorphic Latency
* **The Scientific Gap:** Conventional neurofeedback displays delayed feedback (500–2000 ms), disrupting the biological credit-assignment window of cortical plasticity.
* **Our Solution:** The direct inference loop executes at **10–12 FPS (~80–100 ms per step)** on consumer GPUs (e.g., NVIDIA RTX 3060). This latency matches the duration of a single endogenous biological $\theta$-cycle (~100–160 ms), coupling the brain's internal search phase directly to visual morphing.

---

## 📐 Mathematical Signal Processing Pipeline

### Stage 1: Preprocessing & Global Theta Extraction
Raw bipolar channels are centered using Common Average Referencing (CAR) and notch-filtered at 50 Hz and 100 Hz. The global reference clock $\Phi_\theta(t)$ is computed via the Kuramoto order parameter over all 16 electrodes filtered around $6.0 \pm 1.5\text{ Hz}$:
$$\Phi_\theta(t) = \mathrm{angle}\left( \frac{1}{16} \sum_{c=0}^{15} \frac{Z_{\theta, c}(t)}{|Z_{\theta, c}(t)|} \right)$$
where $Z_{\theta, c}(t)$ is the complex analytic signal of channel $c$.

### Stage 2: Dense Gamma Slicing & von Mises Phase Gating
The 30–85 Hz spectrum is segmented into 32 Gaussian bandpass filters $\mathbf{f}_{\gamma_k}$. Each frequency band is temporally weighted according to its position within the theta cycle:
$$w_k(t) = \frac{\exp(3.2 \cos(\Phi_\theta(t) - \theta_k))}{\sum_\tau \exp(3.2 \cos(\Phi_\theta(\tau) - \theta_k))}, \quad \theta_k = -\pi + \frac{2\pi}{32}\left(k + 0.5\right)$$

The gated cross-spectral phasor matrix for each electrode pair $p = (i, j)$ is:
$$\boldsymbol{\Psi}_k(p) = \sum_{t} \left( P_{\gamma_k, i}(t) \cdot P_{\gamma_k, j}^*(t) \right) w_k(t)$$

### Stage 3: Anchor-Referenced VTE ($i\text{PLV}$)
To eliminate Brownian drift and phase-wrapping, cross-spectral matrices are computed relative to the theta-cycle start anchor $\mathbf{\Psi}_0$ (Past / 30 Hz):
$$i\text{PLV}_k(p) = \Im\left( \mathbf{\Psi}_k(p) \cdot \mathbf{\Psi}_0^*(p) \right) \in \mathbb{R}^{120}$$

### Stage 4: SVD Vocabulary Subspace Projection
To prevent chaotic out-of-distribution latent trajectories, the 120-pair phase vectors are projected onto the top 120 singular vectors $\mathbf{V}_{120} \in \mathbb{R}^{120 \times 768}$ computed via SVD from the complete 49,408-token CLIP vocabulary embedding matrix $\mathbf{E}_{\text{vocab}}$: 
$$\mathbf{h}_k = \frac{\mathrm{iPLV}_k \cdot \mathbf{V}_{120}}{|\mathrm{iPLV}_k \cdot \mathbf{V}_{120}| + \epsilon} \in \mathbb{R}^{768}$$

### Stage 5: Unbiased Attention-Sequence Injection
The 32 temporal slots are interpolated along the 77-token Cross-Attention context length of Stable Diffusion:
$$\mathbf{E}_{\text{drift}} = \mathrm{Interp}_{32 \to 77}(\mathbf{h}_{0..31}) \in \mathbb{R}^{1 \times 77 \times 768}$$
$$\mathbf{E}_{\text{conditioning}} = \mathrm{Encode}(\text{" "}) + \alpha \cdot \frac{\mathbf{E}_{\text{drift}}}{| \mathbf{E}_{\text{drift}} |}$$
where $\text{Encode}(\text{""})$ provides the structural positional encodings of an empty canvas, ensuring **zero text-prompt bias**.

---

## ⚡ System Architecture & Execution Profiles

| Component | Port / Interface | Role | Throughput |
|---|---|---|---|
| **`brain_server.py`** | `localhost:6000` | GPU daemon holding Stable Diffusion LCM + TAESD in VRAM. Receives `prompt_embeds` and performs single-step latent inference. | 10–12 FPS |
| **`semanic_working_memory_sd_lcm.py`** | LSL / Local | Real-time PyTorch CUDA DSP, Kuramoto sync, SVD projection, C++ OpenCV Fast-DCT serialization, Tri-Panel UI. | 200+ FPS |

---

## 🖥️ Tri-Panel Diagnostic Interface

1. **Top-Left (Continuous Latent Walk):** Real-time 1-step diffusion output executing continuous StyleGAN-like geodesic walks across the semantic manifold.
2. **Top-Right (Celestial PCA Concept Radar):** Truncated 2D PCA representation of the 49,408 vocabulary embeddings. The 32-segment Mind Vine glides across conceptual clusters, transitioning from blue ($S_0$, Past) to magenta ($S_{31}$, Future).
3. **Bottom (Spatially Transposed Phase Field):** 100% GPU-rendered continuous phase spectrogram ($X: 0^\circ \to 360^\circ$ physical scalp angle, $Y: 30 \to 85\text{ Hz}$) displaying cortical traveling waves.

---

## 📡 Hardware Specification: FreeEEG16-alpha2

* **Geometry:** Circular PCB, 26 mm outer diameter.
* **Electrode Array:** 16 active dry gold-plated pogo-pin electrodes (~2.7 mm inter-electrode pitch).
* **Reference System:** Integrated local Reference and Ground on the same 26 mm footprint (eliminating long earclip/mastoid wire loops).
* **Sampling Rate:** 250 Hz, 24-bit ADC via Lab Streaming Layer (LSL).

---

## 🛠️ Quick Start

### 1. Start the CUDA Inference Backend:
```bash
python brain_server.py
```

### 2. Start the High-Speed Neurofeedback Client:
```bash
python semanic_working_memory_sd_lcm.py
```
*(If no active LSL stream is detected, the engine falls back to real-time synthetic trajectory simulation).*

---

## <a name="references"></a>📚 References

1. **Lisman, J. E., & Idiart, M. A. (1995).** Storage of 7 +/- 2 short-term memories in oscillatory subcycles. *Science*, 267(5203), 1512–1515.  
   DOI: [10.1126/science.7624776](https://doi.org/10.1126/science.7624776)
2. **Lundqvist, M., Herman, P., & Miller, E. K. (2018).** Working Memory 2.0: Dynamic network interactions. *Neuron*, 100(2), 463–475.  
   DOI: [10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
3. **Lundqvist, M., Rose, J., Herman, P., Brincat, S. L., Buschman, T. J., & Miller, E. K. (2016).** Gamma and beta bursts underlie working memory. *Neuron*, 90(1), 152–164.  
   DOI: [10.1016/j.neuron.2016.02.043](https://doi.org/10.1016/j.neuron.2016.02.043)
4. **Heusser, A. C., Poeppel, D., Ezzyat, Y., & Davachi, L. (2016).** Episodic sequence memory is supported by a theta-gamma phase code. *Nature Neuroscience*, 19(10), 1374–1379.  
   DOI: [10.1038/nn.4314](https://doi.org/10.1038/nn.4314)
5. **Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).** Identifying true brain interaction from EEG data using the imaginary part of coherency. *Clinical Neurophysiology*, 115(10), 2292–2307.  
   DOI: [10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
6. **Bruña, R., Maestú, F., & Pereda, E. (2018).** Phase locking value revisited: teaching new tricks to an old dog. *Journal of Neural Engineering*, 15(5), 056011.  
   DOI: [10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
7. **Redish, A. D. (2016).** Vicarious trial and error. *Nature Reviews Neuroscience*, 17(3), 147–159.  
   DOI: [10.1038/nrn.2015.30](https://doi.org/10.1038/nrn.2015.30)
8. **Stokes, M. G. (2015).** 'Activity-silent' working memory in prefrontal cortex: a dynamic coding framework. *Trends in Cognitive Sciences*, 19(7), 394–405.  
   DOI: [10.1016/j.tics.2015.05.004](https://doi.org/10.1016/j.tics.2015.05.004)
9. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory. *Proceedings of the National Academy of Sciences*, 115(5), 1117–1122.  
   DOI: [10.1073/pnas.1717731115](https://doi.org/10.1073/pnas.1717731115)
10. **Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).** Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. *Human Brain Mapping*, 28(11), 1178–1193.  
    DOI: [10.1002/hbm.20346](https://doi.org/10.1002/hbm.20346)

