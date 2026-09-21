# CardioRT: Engineering & Algorithmic Documentation

## 1. Algorithmic Approach & Problem Formulation

In cardiology interventional fluoroscopy and cine angiography, X-ray beams penetrate thoracic anatomy before hitting a dynamic flat-panel detector. According to the Beer-Lambert law of radiation attenuation:

$$I(x,y) = I_0 \cdot \exp\left(-\int \mu(x,y,z)\,dz\right)$$

The acquired image exhibits a complex superposition of:
1. **Contrast-filled coronary arteries:** Narrow tubular structures ($1-10\text{ px}$ width) filled with radio-opaque iodinated contrast agents ($\mu_{iodine} \gg \mu_{tissue}$).
2. **Dense anatomical bone structures:**
   - **Vertebral spine:** Wide ($30-80\text{ px}$), predominantly vertical attenuation corridor with periodic vertebral body variations.
   - **Rib cage:** Curved, oblique bone bands ($15-40\text{ px}$) crossing the thoracic field at varying angles.
3. **Soft-tissue and lung fields:** Broad, low-frequency density ramps and high-transmittance lung parenchyma.
4. **Quantum noise (photon mottle):** Poisson-distributed noise resulting from low radiation dose constraints in clinical fluoroscopy.

### Optical Density Space Mapping
Direct subtraction or linear filtering in transmission space causes severe non-linear coupling because tissue attenuation is multiplicative. CardioRT maps raw images to the optical density (OD) domain:

$$OD(x,y) = -\ln\left(\frac{I(x,y) + \epsilon}{I_0}\right) = \ln(I_0) - \ln(I(x,y) + \epsilon)$$

In OD space:
$$OD_{total}(x,y) = OD_{coronary}(x,y) + OD_{spine}(x,y) + OD_{ribs}(x,y) + OD_{lungs}(x,y) + \eta(x,y)$$
Background subtraction becomes strictly **additive**, enabling robust anatomical decomposition.

---

## 2. Processing Pipeline Modules

### 2.1 Ingest & Windowing (`cardio_rt.preprocess.windowing`)
- Subsamples the input array on an $8\times 8$ grid to extract robust $0.5\%$ and $99.5\%$ percentiles in $< 0.1\text{ ms}$.
- Detects the lead shutter border using downsampled morphological connected components.

### 2.2 Noise Suppression (`cardio_rt.noise.spatial_denoise`)
- Spatial edge-preserving separable Gaussian filter ($\sigma = 1.0$) combined with high-frequency gating.
- For cine mode: Motion-adaptive IIR temporal filter that gates recursive temporal averaging by pixel-level frame differences, eliminating motion smearing on beating coronary vessels.

### 2.3 Spine Suppression (`cardio_rt.rib_spine.spine_suppression`)
- Exploits the strong vertical orientation of the vertebral column.
- Anisotropic Gaussian smoothing ($\sigma_y \approx 35.0, \sigma_x \approx 7.0$) isolates the spine while leaving branching, non-vertical coronary arteries unaffected.
- Bounded subtraction attenuates vertebral bone density.

### 2.4 Rib Suppression (`cardio_rt.rib_spine.rib_suppression`)
- Evaluates large-sigma Hessian eigenvalues ($\sigma \ge 14\text{ px}$) on a $4\times$ downsampled grid.
- Rib ridges have strong negative cross-sectional curvature at large scales. Thin vessels ($\sigma \le 3\text{ px}$) have negligible response at this scale.
- The resulting bone map is softly subtracted from the optical density map.

### 2.5 Lung Background Suppression (`cardio_rt.lung_bg.lung_suppression`)
- 4-level Laplacian pyramid decomposition separating frequency bands.
- Base residual (DC and low-frequency lung fields) is heavily attenuated (weight $0.12$).
- High-frequency vessel bands are preserved (weight $1.0$).

### 2.6 Multi-Scale Coronary Vesselness (`cardio_rt.vesselness.frangi_fast`)
- For 2D Hessian matrix $H = \begin{bmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{bmatrix}$, closed-form analytical eigenvalues are computed:
  $$\lambda_{1,2} = \frac{(I_{xx} + I_{yy}) \pm \sqrt{(I_{xx} - I_{yy})^2 + 4 I_{xy}^2}}{2}$$
- Frangi tubular vesselness:
  $$\mathcal{V}_\sigma = \begin{cases} 0 & \text{if } \lambda_2 \ge 0 \\ \exp\left(-\frac{\mathcal{R}_B^2}{2\beta^2}\right)\left(1 - \exp\left(-\frac{\mathcal{S}^2}{2c^2}\right)\right) & \text{if } \lambda_2 < 0 \end{cases}$$
- Accelerated via Numba parallel JIT kernel.
- Converted into a **continuous soft gain map** $G(x,y) = 1.0 + \gamma \mathcal{V}(x,y)$, ensuring faint distal branches are boosted without binary thresholding artifacts.

### 2.7 Contrast Enhancement (`cardio_rt.enhance.contrast_enhance`)
- 16-bit CLAHE with grid size $(2, 2)$ to optimize throughput.
- Precomputed 16-bit lookup table (LUT) for instantaneous tone/gamma adjustment ($< 0.8\text{ ms}$).
- Vessel-guided unsharp mask: high-frequency sharpening is gated by $(G(x,y) - 1.0)$, ensuring noise in flat background tissue is never amplified.

---

## 3. Parameter Tuning Guide

| Parameter | Default | Tuning Guidance |
| :--- | :---: | :--- |
| `spine_strength` | 0.75 | Increase to 0.85 for heavily calcified vertebrae; decrease to 0.50 if vessels run vertically. |
| `rib_sigma` | 14.0 | Scale proportionally with detector resolution ($14\text{ px}$ for $512$, $28\text{ px}$ for $1024$). |
| `rib_strength` | 0.65 | Reduce if thin diagonal branches overlap rib edges. |
| `lung_coarse_weight`| 0.12 | Lower (0.05) to eliminate strong lateral lung gradients; raise (0.25) to preserve soft tissue context. |
| `vessel_scales` | (1.5, 3.0) | Add scale 4.5 for dilated coronary aneurysms; use (1.5,) for emergency high-speed mode. |
| `vessel_gain_strength`| 1.50 | Increase to 2.5 for faint sub-optimal contrast injection; decrease to 1.0 if noise is elevated. |
| `clahe_clip_limit` | 2.0 | Standard 2.0; reduce to 1.5 to avoid over-enhancing micro-texture. |

---

## 4. Failure Modes & Mitigations

1. **Extreme Non-Translational Patient Motion:**
   - *Failure:* Severe patient twisting or coughing causes phase correlation misregistration in cine DSA subtraction.
   - *Mitigation:* Single-frame fallback automatically takes precedence when phase correlation peak response $< 0.15$.
2. **Dense Metallic Implants (Pacemaker Leads, Sternal Wires):**
   - *Failure:* High atomic number metals produce extreme star artifacts and shadow blooms.
   - *Mitigation:* Collimator / threshold mask flags extreme radiopaque objects to prevent histogram skew.
3. **Contrast Agent Washout (End of Injection):**
   - *Failure:* Diminishing iodine concentration reduces Hessian eigenvalues.
   - *Mitigation:* Soft gain mapping ensures smooth attenuation without sudden vessel drop-out.
