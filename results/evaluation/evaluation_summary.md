# Quantitative Evaluation Report: CardioRT Performance

### Key Quantitative Metrics Across Clinical Scenarios
| Scenario | CNR Raw | CNR Enhanced | CNR Boost | BG Suppression Ratio | Clutter Reduction % | Vessel SSIM | Dice Score | Noise Reduction % | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy Case (High Contrast)** | 2.17 | **4.74** | **2.18x** | **1.04x** | 8.0% | **0.563** | **0.154** | **-100.0%** | 26.3 ms |
| **Average Case (Nominal Cine)** | 2.14 | **4.79** | **2.24x** | **1.04x** | 8.0% | **0.564** | **0.151** | **-100.0%** | 26.35 ms |
| **Hard Case (Low Contrast + High Noise)** | 1.4 | **4.83** | **3.45x** | **1.05x** | 9.2% | **0.465** | **0.142** | **-100.0%** | 26.45 ms |