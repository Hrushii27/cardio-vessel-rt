# CardioRT: Datasets & Ingestion Reference

## 1. Synthetic Cine Dataset (Primary Development & Benchmarking)

CardioRT includes a physics-based procedural generator (`scripts/generate_synthetic.py`) producing true 16-bit fluoroscopy sequences with ground-truth anatomical masks.

### Simulated Clinical Effects
- **Branching Coronary Tree:** Procedurally generated Left Coronary Artery tree modeling LMCA, proximal/mid/distal LAD with D1/D2 diagonal branches, and LCx with OM1 obtuse marginal branches.
- **Physical Contrast Inflow:** Bolus progression along the vascular tree following Murray's law of daughter branch diameter tapering.
- **Dual Physiological Motion:**
  - Fast cardiac contraction/torsion cycle ($\sim 75\text{ bpm}$, $\sim 1.25\text{ Hz}$).
  - Slow respiratory thoracic elevation/descent ($\sim 15\text{ bpm}$, $\sim 0.25\text{ Hz}$).
- **Overlaid Anatomy:** Vertebral spine column with periodic disc modulation, 4 oblique curved rib bands, radiolucent lung fields, diaphragm dome, and radio-opaque catheter.
- **Noise Dynamics:** True Poisson photon-counting quantum mottle + Gaussian detector readout noise.

### Committed Sample Data (`sample_data/`)
- `sample_data/cine_512_standard/`: 30 frames (512x512, 16-bit PNG) with binary ground-truth vessel masks.
- `sample_data/cine_512_hard/`: 30 frames (512x512, low contrast injection, high quantum noise, tachycardia motion).
- `sample_data/cine_1024_standard/`: 10 frames (1024x1024, true 16-bit PNG) for high-resolution testing.

---

## 2. Clinical Datasets Guide

CardioRT provides automated helpers and conversion tools in `scripts/download_datasets.py` for standard clinical research datasets.

### 1. ARCADE (MICCAI 2023)
- **Title:** Automated Segmentation and Quantification of Coronary Artery Disease Challenge
- **Source:** [Zenodo Record 7949216](https://zenodo.org/record/7949216)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Popov et al., "ARCADE: Challenge on Automated Segmentation and Quantification of Coronary Artery Disease", MICCAI 2023.
- **Native Bit Depth:** 8-bit grayscale.
- **Ingestion Note:** When converted using `scripts/download_datasets.py`, images are scaled to 16-bit (`val * 257`) and flagged as `"was_upscaled_from_8bit": true`.

### 2. XCAD
- **Title:** X-ray Coronary Angiography Dataset
- **Source:** [GitHub Repository](https://github.com/ashispati/XCAD)
- **License:** Non-commercial academic research use.
- **Citation:** Pati et al., "Coronary Artery Segmentation in Angiograms using Deep Learning", 2020.
- **Native Bit Depth:** 8-bit.

### 3. DCA1
- **Title:** Database of Coronary Angiograms (Cervantes-Sanchez et al.)
- **Source:** [GitHub Repository](https://github.com/focvs/DCA1)
- **License:** Open Access Research License.
- **Citation:** Cervantes-Sanchez et al., Computer Methods and Programs in Biomedicine, 2019.
- **Native Bit Depth:** 8-bit.

### 4. CADICA
- **Title:** Coronary Artery Disease in Invasive Coronary Angiography
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/p9bpx9ctcv/1)
- **License:** CC BY 4.0
- **Citation:** Ortiz et al., Mendeley Data, 2023.
- **Data Type:** Invasive coronary angiography cine video sequences.

---

## 3. Provenance & Bit-Depth Policy

To maintain scientific integrity:
1. **Never Claim False 16-bit Precision:** Upscaled 8-bit clinical datasets are explicitly flagged in accompanying metadata JSON files.
2. **End-to-End float32 / uint16 Precision:** The internal pipeline operates strictly with float32 buffers and outputs native uint16 tensors. Down-quantization to uint8 occurs only during GUI/display rendering.
