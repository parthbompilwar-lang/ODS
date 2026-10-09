# Computer Vision Based Semi-Automated Offside Detection System (ODS)

A broadcast-grade, FIFA Law 11-compliant Computer Vision Offside Detector engineered with **YOLOv11**, perspective pitch geometry, and automated Video Assistant Referee (VAR) graphics rendering.

Tested and validated on real-world broadcast footage from the **Offside Detection Dataset** (482 high-resolution match frames).

---

## 🌟 Key Features

1. **Pitch & Ground Condition Invariance**:
   - Works across varied conditions: floodlights, daylight, harsh shadows, weathered pitch markings, and mud.
   - **Triple-redundant geometry engine**:
     - *Mode 1 (AI Edge & Hough)*: Automatic vanishing point estimation of goal-parallel lines.
     - *Mode 2 (4-Point Metric Homography)*: Direct camera-to-pitch homography matrix $H$ mapping pixels to metric meters ($105\text{m} \times 68\text{m}$).
     - *Mode 3 (Interactive Fine-Tuning)*: Real-time slider calibration on the VAR review dashboard.

2. **FIFA Law 11 Rules Engine**:
   - Determines offside strictly from **playable body parts** (Head, Shoulders, Torso, Knees, Feet).
   - **Arms and hands are explicitly filtered out** from both attacking and defending lines.
   - Identifies the **second-last opponent** as the defensive baseline.
   - Checks the **ball line**: if the ball is nearer to the opponent goal than the second-last defender, the ball line defines the offside baseline.
   - Halfway line constraint: players in their own half cannot be offside.

3. **Premier League Style VAR Broadcast Overlays**:
   - **Virtual Pitch Lines**: Calibrated perspective lines (Blue for defender baseline, Red for attacker leading line).
   - **3D Vertical Drop Lines**: Vertical lines dropped from elevated playable parts (shoulder/head) down to the pitch plane with circular anchors.
   - **Translucent Shaded Offside Zone**: Darkened pitch polygon extending beyond the offside line towards the goal line.
   - **Top-Down 2D Pitch Radar**: Real-time tactical radar minimap showing true player distributions and offside line.
   - **VAR Decision Banners**: High-visibility decision badge (`OFFSIDE` / `ONSIDE`) with margin measurement (e.g. `+0.28m`).

4. **Deep Learning Object Detection & Pose**:
   - YOLOv11 / YOLOv8 state-of-the-art keypoint detection and player localization.
   - Unsupervised CIE-Lab color clustering for automatic team separation (Team A, Team B, Goalkeeper, Referee).

---

## 🚀 Quickstart Guide (Video Input - No Streamlit)

### 1. Process Any Match Video Directly
You can run offside detection on any input video file (`.mp4`, `.avi`, `.mov`, etc.):
```bash
py -3.11 process_video.py --video path/to/your_video.mp4 --output var_output.mp4 --direction left --team 0
```
- `--video`: Path to your input football video file.
- `--output`: Path where the annotated VAR broadcast video will be saved.
- `--direction`: Attacking direction towards opponent goal (`left` or `right`).
- `--team`: Attacking team index (`0` or `1`).
- `--model`: Path to trained weights (defaults to `models/yolo11_offside_best.pt`).

### 2. Interactive Video Review Player (OpenCV Window)
To watch the video with real-time VAR overlays and pause/step frame-by-frame:
```bash
py -3.11 process_video.py --video sample_match.mp4 --display
```
**Interactive Keyboard Controls:**
- `[SPACE]`: Pause / Resume playback.
- `[D] / [Right Arrow]`: Step forward 1 frame (perfect for freezing on the pass moment).
- `[A] / [Left Arrow]`: Step backward 1 frame.
- `[T]`: Toggle attacking team ($0 \leftrightarrow 1$).
- `[G]`: Toggle goal direction ($Left \leftrightarrow Right$).
- `[S]`: Save high-resolution freeze-frame VAR decision snapshot.
- `[Q] / [ESC]`: Exit and save video.

### 3. Run Sample Demo Script
To generate sample VAR broadcast decisions on match stills:
```bash
py -3.11 run_demo.py
```
Generated frames will be saved to `e:\ODS\demo_results/`.

### 4. Run Dataset Benchmark
To evaluate the system across the 482-image dataset:
```bash
py -3.11 evaluate.py
```

---

## 📊 Dataset & Research References

This system builds upon and enhances:
1. **IEEE 2025**: *Implementation of YOLOv11 for Automatic Offside Detection in Football* (Ayu Nur Indahsari et al.)
2. **ACM MMSports 2020**: *A Dataset & Methodology for Computer Vision based Offside Detection in Soccer* (Neeraj Panse & Ameya Mahabaleshwarkar)
   - Dataset repository: [https://github.com/Neerajj9/Computer-Vision-based-Offside-Detection-in-Soccer](https://github.com/Neerajj9/Computer-Vision-based-Offside-Detection-in-Soccer)
   - 482 curated broadcast match scenes with ground-truth player keypoints and team assignments.
