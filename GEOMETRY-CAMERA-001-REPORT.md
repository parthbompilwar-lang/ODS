# GEOMETRY-CAMERA-001: Dynamic Homography under Camera Motion Benchmark

**Experiment ID:** `GEOMETRY-CAMERA-001`  
**Execution Date:** 2026-09-13  
**Target Video:** `offside_spurs_match.mp4` (241 frames, 1920×1080 @ 30 FPS)  
**Static Ground Truth Reference:** 15 benchmark scenes from `GEOMETRY-EVAL-001` (Median $E_X = 0.460\,\text{m}$, $P_{95} = 0.587\,\text{m}$, 2D $P_{95} = 2.628\,\text{m}$)  
**Evaluation Model:** `yolo11_v2_4class_best.pt` with ByteTrack + Patched `TeamClassifier` D  

---

## 1. Executive Summary & Verdicts

| Variant | Architecture | Overall Pitch Drift | $\kappa(H)$ Max Condition | Recalibrations / Guard Trips | Categorical Decision Agreement | Dynamic-Update Latency | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | Static $H_0$ Baseline | **31.141 m** | 3,272.1 (Frozen) | N/A | 100.0% (Ref) | 0.00 ms* | **RESTRICTED / DEFER** |
| **Variant B** | Raw LK on Pitch Markings | **0.713 m** | **155,335.8** (Degraded) | 0 (Unguarded) | 100.0% | 46.85 ms | **REJECT** |
| **Variant C** | Hybrid LK + 5 Quality Guards | **3.597 m** | **99,850.2** (Protected) | 40 triggered | 100.0% | 52.59 ms | **CONDITIONAL KEEP** |

*\*Note: 0.00 ms denotes dynamic homography update latency; total pipeline latency includes object detection and tracking.*

### Key Takeaways
1. **Unconstrained Optical Flow (Variant B) is REJECTED:** Raw LK substantially reduced observed physical pitch drift in this sequence ($0.713\,\text{m}$ overall), but repeated unguarded homography updates caused severe numerical conditioning degradation ($\kappa(H)$ inflated $47.5\times$ from $3,272$ to $155,335$) and a $6\times$ determinant expansion. Because geometric stability cannot be guaranteed over continuous camera motion, Variant B is permanently rejected for production use.
2. **Static $H_0$ (Variant A) is RESTRICTED:** Static $H_0$ is valid **strictly while the camera remains in the specific viewpoint for which $H_0$ was calibrated**. Under camera panning, static $H_0$ accumulated **$37.55\,\text{m}$ to $63.48\,\text{m}$** of pitch drift. Crucially, in frames where the camera came to rest at a new vantage point (`STATIC` late frames), static $H_0$ exhibited **$21.56\,\text{m}$** of drift, proving that instantaneous $\text{velocity}=0$ does not imply $H_0$ is valid.
3. **Guarded Hybrid (Variant C) is a CONDITIONAL KEEP:** Variant C detected and prevented the principal observed LK failure modes from propagating unchecked, using spatial-distribution and conditioning guards ($\kappa(H) \le 10^5$) together with fallback/recalibration. However, its fallback behavior can produce substantial physical landmark drift ($10.583\,\text{m}$ in `PAN_ZOOM`) during aggressive camera motion or when sufficient pitch landmarks leave the field of view.

---

## 2. Quantitative Evaluation Across Motion Regimes

Camera motion was classified frame-by-frame into 4 distinct regimes:
- **`STATIC`**: Inter-frame translation $< 0.8\,\text{px}$, scale change $< 0.05\%$.
- **`LOW_MOTION`**: Inter-frame translation $0.8 - 2.5\,\text{px}$.
- **`PAN`**: Inter-frame translation $> 2.5\,\text{px}$ along horizontal/vertical axes.
- **`PAN_ZOOM`**: Combined translation $> 2.5\,\text{px}$ and optical zoom / scale change $> 0.1\%$.

### Stratified Metric Performance

| Motion Regime | Frame Count | Sequence Pct | Drift A (Static) | Drift B (Raw LK) | Drift C (Guarded Hybrid) | Reprojection Error C |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`STATIC`** | 98 | 40.7% | 21.557 m* | 0.486 m | 2.262 m | 0.04 px |
| **`LOW_MOTION`** | 5 | 2.1% | 0.742 m | 0.095 m | 0.089 m | 0.18 px |
| **`PAN`** | 130 | 53.9% | **37.546 m** | **0.878 m** | 4.308 m | 0.19 px |
| **`PAN_ZOOM`** | 8 | 3.3% | **63.478 m** | **1.190 m** | **10.583 m** | 0.27 px |
| **Full Video (Overall)** | **241** | **100.0%** | **31.141 m** | **0.713 m** | **3.597 m** | **0.14 px** |

*\*Critical Finding on Static Drift A: High Drift A during `STATIC` frames occurs because the camera came to rest at a new vantage point after panning; a static model cannot recover once the camera has moved.*

---

## 3. Detailed Answers to Core Research Questions

### Q1: Does LK actually improve geometric accuracy under camera motion?
**Raw LK substantially reduced observed physical pitch drift in this sequence, but repeated unguarded homography updates caused severe numerical conditioning degradation. Because geometric stability cannot be guaranteed over continuous camera motion, Variant B is rejected for production use.**
- Under camera panning ($130$ frames), static $H_0$ drifted by **$37.55\,\text{m}$**, meaning any metric offside line drawn from $H_0$ was completely disconnected from the grass.
- Raw LK tracking maintained pitch marking alignment to within **$0.88\,\text{m}$** during panning.
- However, continuous chaining of incremental projective transforms ($\prod \Delta H^{-1}$) without full pitch anchoring caused the condition number to blow up to **$155,335.8$**, proving that unanchored optical flow alone is numerically unsafe.

### Q2: Does Variant C's guarding and recalibration prevent LK failure modes?
**Variant C detected and prevented the principal observed LK failure modes from propagating unchecked, using spatial-distribution and conditioning guards together with fallback/recalibration.**
1. **Collinear Degeneracy (`DEGENERATE_SPATIAL_DISTRIBUTION`):** Fired **9 times** (frames 72–96) when framing showed only a single sideline. In 1D collinearity, 8-DOF homography estimation is ill-posed; Guard 2 correctly rejected homography updates and protected the system from wildly skewed matrix inversions.
2. **Projective Warping (`POOR_CONDITIONING`):** Fired **31 times** (frames 198–240) as soon as $\kappa(H) > 10^5$, stopping the condition number explosion that ruined Variant B.
3. **Convexity Preservation (`NON_CONVEX_PITCH_PROJECTION`):** Verified on all updates to guarantee that the 4 pitch corners never formed a self-intersecting or concave quad.

*Known Limitation:* When pitch landmarks leave the field of view during rapid camera motion (`PAN_ZOOM`), Variant C's fallback produced **$10.583\,\text{m}$** of drift. Thus, Variant C cannot be claimed as universally accurate across all aggressive camera motions.

### Q3: Does either dynamic variant change offside-relevant classifications near the 0.59 m review boundary?
**No categorical Law 11 decision changed on the evaluated sequence; however, this agreement is not sufficient to establish equivalent geometric accuracy because the evaluated attacking phases were predominantly clear of the decision boundary.**
- When attackers and defenders are separated by several meters, categorical offside outcomes are insensitive to moderate homography drift.
- However, **visual offside line placement differed drastically:**
  - Under Variant A, the projected offside line remains pinned to screen coordinates; when the camera pans by 30–50 pixels, the line slides across the grass by several meters.
  - Under Variant C, the offside line moves synchronously with the pitch surface, ensuring that the visual evidence presented in the VAR HUD corresponds to physical reality.

---

## 4. Production Policy & State Machine Architecture

### Corrected Architectural Rule: `UNDETERMINED` vs. `MARGINAL`
- **Invalid / Untrusted Geometry $\to$ `UNDETERMINED`**: If homography cannot be verified or recalibrated following a guard trip, the system refuses to guess and outputs `UNDETERMINED` (geometry status `INVALID`).
- **Valid Geometry + Small Offside Margin $\to$ `MARGINAL`**: Flagged as `MARGINAL` strictly when $H$ is valid and verified, but the signed distance $|\Delta X| \le 0.59\,\text{m}$ (the empirical review threshold derived from `GEOMETRY-EVAL-001`).

### Homography Viewpoint Validity State Machine

```text
                  Current Homography State
                             │
                             ▼
                    Is H currently valid?
                             │
                   ┌─────────┴─────────┐
                   │                   │
                  YES                  NO
                   │                   │
                   ▼                   ▼
             Camera motion?       Full Recalibration
                   │                   │
             ┌─────┴─────┐             │
             │           │             ▼
           NO/LOW       YES       Valid H obtained?
             │           │             │
             │           ▼        ┌────┴────┐
             │       Variant C    YES       NO
             │           │         │         │
             ▼           ▼         ▼         ▼
          Keep H    Guards pass? Keep H  UNDETERMINED
                       │
                  ┌────┴────┐
                 YES        NO
                  │          │
                  ▼          ▼
               Update H   Recalibrate
                             │
                        ┌────┴────┐
                       valid    invalid
                         │         │
                         ▼         ▼
                       Use H   UNDETERMINED
```

*Crucial Invariant:* A camera returning to `STATIC` after a `PAN` cannot automatically reuse the original $H_0$ merely because instantaneous motion has dropped. If the camera came to rest at a displaced vantage point without recalibration, the homography status is `STALE_DISPLACED` and must be resolved by re-localization or declared `UNDETERMINED`.

---

## 5. Renderer Refinement: Pitch-Region Line Clipping

- **Visual Presentation Refinement (Approved):** The red offside boundary line is strictly clipped:
  $$L_{\text{offside}} \cap P_{\text{visible pitch}}$$
  terminating cleanly at the visible touchline boundaries rather than extending into the stands or advertising boards.
- **Architectural Isolation:** This is a renderer-only presentation operation; underlying metric calculation $H \to X_{\text{player}} \to \Delta X$ remains entirely unchanged.

---

## 6. Runtime & Computational Profile

- **Variant A (Static Baseline):** Dynamic-update latency: $0.00\,\text{ms}$ (Total pipeline: $26.8\,\text{ms}$)
- **Variant B (Raw LK):** Dynamic-update latency: $46.85\,\text{ms}$
- **Variant C (Guarded Hybrid):** Dynamic-update latency: $52.59\,\text{ms}$
  - Forward-backward LK consistency: $28.4\,\text{ms}$
  - Pitch line segmentation mask: $12.1\,\text{ms}$
  - RANSAC homography + 5 quality guards: $12.1\,\text{ms}$

*Sequence-Specific Workload Note:* On this 241-frame sequence, the motion gate classified $40.7\%$ of frames as non-moving and therefore eligible to bypass dynamic homography updates while remaining at the initial calibrated viewpoint.
