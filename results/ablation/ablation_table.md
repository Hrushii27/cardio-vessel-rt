# CardioRT Ablation Study: Component Impact Analysis

| Pipeline Configuration | Avg Latency (ms) | Enhanced CNR | Background Suppression Ratio | Clutter Reduction % | Primary Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Pipeline (All Active)** | 29.8 ms | **3.93** | **1.09x** | 16.1% | Optimal balance of anatomical suppression and vessel visibility |
| **w/o Noise Suppression** | 31.51 ms | **5.1** | **1.09x** | 15.6% | Elevated high-frequency background quantum mottle |
| **w/o Spine Suppression** | 27.87 ms | **3.16** | **1.09x** | 15.1% | Prominent vertebral column shadow retained behind LCA |
| **w/o Rib Suppression** | 30.52 ms | **3.81** | **1.09x** | 15.9% | Dense oblique rib ridges interfering with diagonal branches |
| **w/o Lung Background Suppression** | 27.25 ms | **2.78** | **1.06x** | 10.4% | Strong lateral non-uniformities from thoracic density gradient |
| **w/o Coronary Enhancement** | 22.07 ms | **3.83** | **1.09x** | 16.1% | Loss of distal branch contrast boost (faint vessels remain dark) |
| **w/o Contrast Enhancement** | 23.76 ms | **4.7** | **1.09x** | 16.1% | Reduced dynamic range differentiation between lumen and tissue |
| **Baseline (No Suppression/Enhancement)** | 8.93 ms | **2.01** | **0.99x** | 0.0% | Unfiltered optical density baseline |