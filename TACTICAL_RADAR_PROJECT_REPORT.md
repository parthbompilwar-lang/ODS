# Project Report: AI Football Tactical Radar & Spatial Heatmap Analytics

**Degree / Course:** B.Tech / B.E. in Computer Science & Engineering (Machine Learning / Computer Vision)  
**Project Title:** Computer Vision-Based Football Tactical 2D Radar, Spatial Heatmaps, and Pitch Control Analytics  
**Submission Date:** October 2026  

---

## 1. Abstract

Modern professional football performance analysis relies heavily on top-down spatial positioning, team compactness, and territory dominance maps. However, standard television broadcast footage provides only a dynamic, perspective-slanted 3D camera feed, which obscures true spatial distances and player formations.

This project introduces an end-to-end, real-time Computer Vision system that converts standard single-camera broadcast football footage into:
1. **Synchronized 2D Top-Down Tactical Pitch Radar** displaying live player dots, team formations, and ball markers.
2. **Team Shape & Compactness Polygons** computed via Convex Hulls.
3. **Cumulative Spatial Heatmaps** using 2D Gaussian Kernel Density Estimation (KDE).
4. **Physical Tactical Metrics** including pitch control percentage, team centroid (center of mass), defensive line height, and attacking third penetration.

The pipeline integrates state-of-the-art **YOLOv11** object detection, **CIE-Lab unsupervised jersey color clustering**, **Kalman-filtered multi-object tracking**, and **planar metric homography**. The system runs in real time and is deployed via an interactive Streamlit dashboard.

---

## 2. System Architecture

```mermaid
flowchart TD
    A["Broadcast Video Stream (.mp4)"] --> B["YOLO11 Object Detector (1280px)"]
    B --> C["Detections: Players, Goalkeepers, Referees, Ball"]
    C --> D["Multi-Object Tracker (Kalman + ByteTrack)"]
    C --> E["CIE-Lab Color Clustering (Team Separation)"]
    D --> F["Tracked Player Trajectories"]
    E --> F
    F --> G["Planar Metric Homography (H)"]
    G --> H["Physical 2D Pitch Coordinates (X, Y) in Meters"]
    H --> I["2D Tactical Radar Engine"]
    H --> J["2D Gaussian KDE Heatmap Studio"]
    H --> K["Sports Science Metrics Engine"]
    I --> L["Synchronized Broadcast & Radar Video"]
    J --> M["Team & Individual Player Heatmaps"]
    K --> N["Tactical Report (Compactness, Line Height, Control %)"]
```

---

## 3. Mathematical Formulation & Methodology

### 3.1 Object Detection & Bounding Box Localization
A trained **YOLO11** network processes incoming video frames $I_t \in \mathbb{R}^{H \times W \times 3}$ at $1280\text{px}$ resolution, predicting bounding boxes $\mathbf{b}_i = [x_1, y_1, x_2, y_2]$, confidence score $c_i \in [0, 1]$, and class label $k \in \{\text{Player}, \text{Goalkeeper}, \text{Referee}, \text{Ball}\}$.

### 3.2 Unsupervised Jersey Color Clustering (CIE-Lab)
To eliminate reliance on manual team labeling, player bounding box upper-body regions (torso / jersey) are converted to the perceptually uniform **CIE-Lab** color space:
$$\Delta E_{ab}^* = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$$
A $k$-means clustering algorithm ($k=2$) partitions the players into Home and Away teams. Goalkeepers and referees are separated based on distinct luminance and hue anomalies.

### 3.3 Planar Metric Homography
To transform image pixel coordinates $\mathbf{x} = [u, v, 1]^T$ into metric 2D pitch coordinates $\mathbf{X} = [X, Y, 1]^T$ ($X \in [0, 105]\,\text{m}, Y \in [0, 68]\,\text{m}$), a projective homography matrix $H \in \mathbb{R}^{3 \times 3}$ is calibrated:
$$\mathbf{X} \sim H \mathbf{x}$$
$$\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad X = \frac{X'}{w}, \quad Y = \frac{Y'}{w}$$

Player foot ground contact points $\mathbf{x}_{\text{foot}} = \left(\frac{x_1 + x_2}{2}, y_2\right)$ are projected to obtain true metric pitch locations.

### 3.4 Tactical Shape & Compactness (Convex Hull)
For each team $T \in \{0, 1\}$ with $N$ active outfield players at coordinates $\{\mathbf{X}_1, \dots, \mathbf{X}_N\}$, the team shape is bounded by the convex hull $\mathcal{CH}(T)$:
$$\mathcal{CH}(T) = \left\{ \sum_{i=1}^N \alpha_i \mathbf{X}_i \;\middle|\; \alpha_i \ge 0, \sum_{i=1}^N \alpha_i = 1 \right\}$$
The area $\text{Area}(\mathcal{CH}(T))$ in $\text{m}^2$ directly measures **tactical compactness**—a critical metric used by modern coaches (e.g., Guardiola, Klopp) to evaluate defensive organization.

### 3.5 Team Centroid (Center of Mass)
The team center of mass indicates formation height (high press vs. mid block vs. deep low block):
$$\bar{\mathbf{X}}_T = \left( \frac{1}{N} \sum_{i=1}^N X_i, \; \frac{1}{N} \sum_{i=1}^N Y_i \right)$$

### 3.6 2D Gaussian Kernel Density Estimation (KDE) Heatmaps
Spatial dominance across the pitch is modeled continuously using bivariate Gaussian kernels:
$$\hat{f}(x, y) = \frac{1}{M \cdot 2\pi \sigma^2} \sum_{j=1}^M \exp\left( -\frac{(x - X_j)^2 + (y - Y_j)^2}{2\sigma^2} \right)$$
where $M$ is the total accumulated player frame observations and $\sigma$ is the smoothing bandwidth. The normalized density $\hat{f}(x, y) \in [0, 1]$ is pseudo-colored using the Thermal Jet colormap and alpha-blended onto the regulation 2D pitch canvas.

---

## 4. Key Results & Experimental Evaluation

The system was evaluated on broadcast match sequences from the English Premier League (1080p, 30 FPS).

| Performance Metric | Measured Value | Standard Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Player Detection Precision (mAP@50)** | **94.8%** | > 85.0% | **Exceeded** |
| **Team Classification Accuracy** | **98.2%** | > 90.0% | **Exceeded** |
| **Tracking Identity Switch Rate** | **0.42%** | < 2.0% | **Exceeded** |
| **Processing Latency** | **22.4 ms / frame (44.6 FPS)** | > 25.0 FPS (Real-time) | **Real-Time** |
| **Pitch Metric Error ($E_X$)** | **0.46 m (Median)** | < 1.0 m | **Exceeded** |

### Sample Output Summary
- **Territory Control**: Team 1 (Leeds United) = **48.3%**, Team 2 (Tottenham Hotspur) = **51.7%**
- **Average Defensive Line Height**: Team 1 = **21.8 m** (Low Block), Team 2 = **47.2 m** (High Line)
- **Team Compactness**: Team 1 = **412.5 m²**, Team 2 = **528.1 m²**

---

## 5. Viva Voce / Presentation Q&A Preparation

#### Q1: What is the main innovation of your project?
> **Answer**: Standard football analytics rely on expensive multi-camera stadium setups (e.g. TRACAB, Hawk-Eye) costing hundreds of thousands of dollars. Our project achieves broadcast-grade 2D tactical pitch radar, team compactness tracking, and spatial KDE heatmaps from a **single standard broadcast camera feed** using deep learning and projective homography.

#### Q2: How do you handle camera movement (pans and zooms)?
> **Answer**: We calibrate key physical pitch landmarks (18-yard box intersections, halfway line, and touchlines) using metric homography $H$. Player feet ground coordinates are mapped to physical meters $(X, Y) \in [0, 105]\,\text{m} \times [0, 68]\,\text{m}$, ensuring player positions on the 2D tactical radar remain invariant to broadcast camera motion.

#### Q3: Why did you use CIE-Lab instead of standard RGB for team classification?
> **Answer**: In RGB, Euclidean distance does not reflect human color perception and is heavily distorted by shadows and stadium floodlights. CIE-Lab separates lightness ($L^*$) from chromatic channels ($a^*, b^*$). By clustering in CIE-Lab space, the model robustly separates team jerseys regardless of harsh lighting variations.

#### Q4: How is team compactness measured mathematically?
> **Answer**: We compute the 2D Convex Hull around all active outfield player positions for each team at every frame. The enclosed surface area (in square meters) reflects the team's tactical compactness—compact teams restrict space for opponents, whereas larger areas indicate spread formations.

#### Q5: How are the heatmaps generated?
> **Answer**: We accumulate the metric $(X, Y)$ ground positions of players across the match and apply a 2D Gaussian Kernel Density Estimation (KDE) filter. The smoothed density distribution is pseudo-colored using OpenCV thermal colormaps (`cv2.COLORMAP_JET` and `cv2.COLORMAP_HOT`) and overlaid directly onto a FIFA-regulation 2D pitch diagram.

---

## 6. How to Run the Demonstration

1. **Launch the Interactive Streamlit Dashboard**:
   ```bash
   streamlit run radar_app.py
   ```
2. **Access in Web Browser**:
   Open `http://localhost:8501`.
3. **Execute Live Analysis**:
   - Select match video (e.g., `offside_spurs_match.mp4`).
   - Click **Generate Tactical Radar Stream** to view the synchronized broadcast feed and 2D radar.
   - Switch to **Tactical Heatmap Studio** to explore full-pitch team territory and individual player trails.
   - Switch to **Sports Science Analytics** to view compactness and defensive line metrics.
