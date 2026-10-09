# UNCERTAINTY-POLICY-AUDIT-001: Empirical Geometry & Boundary Policy Audit

**Audit Target:** Law 11 Offside Decision Pipeline & Uncertainty Propagation  
**System:** ODS Football Offside Detection  
**Audit Date:** 2026-09-12  
**Benchmark Reference:** `GEOMETRY-EVAL-001` ($E_X$ Median=$0.460\text{ m}$, $P_{95}=0.587\text{ m}$)  

---

## 1. Executive Summary

This audit inspects the complete operational path from raw camera coordinates to the final broadcast decision graphic in `app.py`, `src/engine/offside_engine.py`, `src/engine/play_state.py`, and `src/engine/evidence_snapshot.py`.

### Core Question:
> *Is the empirical calibration uncertainty ($E_X \le 0.587\text{ m} \approx 0.59\text{ m}$) actively propagated into the live runtime decision and broadcast presentation, or does the live application rely solely on the $0.05\text{ m}$ computational level-tolerance?*

### Headline Finding:
> **The live runtime pipeline is decoupled from the uncertainty policy.**  
> While `OffsideEngine` calculates a theoretical `boundary_status` (`MARGINAL` vs. `DECISIVE`), it uses an obsolete holdout threshold ($0.27\text{ m}$). More critically, `boundary_status` is **dropped entirely** when instantiating `EvidenceSnapshot`, ignored by `PlayStateMachine`, and never displayed in `app.py`. The live system currently executes:
> $$\text{Geometry} \longrightarrow \text{Raw } \Delta X \longrightarrow 0.05\text{ m Level Tolerance} \longrightarrow \text{Binary OFFSIDE / ONSIDE}$$
> rather than the intended multi-tiered review policy.

---

## 2. Component-by-Component Trace

### A. `src/engine/offside_engine.py`
1. **Error Budget Constant (Line 20)**:
   ```python
   LEVEL_TOLERANCE_M = 0.05
   DEFAULT_EMPIRICAL_ERROR_BUDGET_M = 0.27  # Derived from GEOMETRY-DEBUG-002 hold-out on 0.jpg
   ```
   *Status*: **STALE**. Still hardcodes $0.27\text{ m}$ from the single-scene test (`0.jpg`), despite `GEOMETRY-EVAL-001` producing cross-scene $P_{95} = 0.587\text{ m} \approx 0.59\text{ m}$.

2. **Binary Decision Gate (Lines 288, 313-314)**:
   ```python
   if in_opponent_half and signed_margin_m > LEVEL_TOLERANCE_M:  # LEVEL_TOLERANCE_M = 0.05
       result.offside_players.append(att)
   ...
   if len(result.offside_players) > 0:
       result.decision = "OFFSIDE"
   else:
       result.decision = "ONSIDE"
   ```
   *Status*: The primary `decision` attribute (`"OFFSIDE"` vs. `"ONSIDE"`) is strictly determined by whether $\Delta X > 0.05\text{ m}$.

3. **Boundary Status Calculation (Lines 323-326, 364-367)**:
   ```python
   if abs(result.margin_val) <= self.empirical_error_budget_m:
       result.boundary_status = "MARGINAL"
   else:
       result.boundary_status = "DECISIVE"
   ```
   *Status*: `result.boundary_status` is populated on `OffsideResult`, but only used to append a text note to `result.explanation`.

---

### B. `src/engine/evidence_snapshot.py`
```python
class EvidenceSnapshot:
    def __init__(
        self,
        frame_index: int, image: np.ndarray, decision: str,
        margin_val: float, margin_unit: str, confidence: float,
        ball_evidence_state: str, offside_line_endpoints, attacker_line_endpoints,
        passer_track_id, offside_boundary_world_x, explanation: str,
        homography_valid: bool, law11_state: str = "OFFSIDE_POSITION",
        receiver_id = None
    ):
```
*Status*: **DATA LOSS**. `EvidenceSnapshot` has **no field** for `boundary_status`, `empirical_error_budget_m`, or `geometry_status`.

---

### C. `src/engine/play_state.py` (Lines 70-84)
```python
self.active_snapshot = EvidenceSnapshot(
    frame_index=contact_f,
    image=self.frozen_image,
    decision=self.last_offside_result.decision,
    margin_val=self.last_offside_result.margin_val,
    margin_unit=self.last_offside_result.margin_unit,
    ...
    # boundary_status is NOT passed
)
```
*Status*: `PlayStateMachine` ignores `boundary_status`. An incident with margin $+0.06\text{ m}$ transitions to `OFFSIDE_OFFENCE` and freezes playback identically to a $+3.50\text{ m}$ breakaway.

---

### D. `app.py` (Live Streamlit Application & Broadcast Renderer)
1. **Decision Graphic (Lines 338-349)**:
   ```python
   if active_snap.decision == "UNDETERMINED":
       dec_title = "DECISION: UNDETERMINED"
   elif active_snap.decision == "OFFSIDE":
       state_lbl = getattr(active_snap, 'law11_state', 'OFFSIDE_POSITION')
       if state_lbl == "OFFSIDE_OFFENCE":
           dec_title = "DECISION NO GOAL - OFFSIDE"
       else:
           dec_title = "VAR: OFFSIDE POSITION DETECTED"
   else:
       dec_title = "DECISION - GOAL / ONSIDE"
   ```
2. **Technical Telemetry Sub-pill (Lines 357-358)**:
   ```python
   sub_txt = f"CONTACT #{active_snap.frame_index} | MARGIN: +{active_snap.margin_val:.2f}{active_snap.margin_unit} | ATTACK: TEAM {attack_team_id} ({dir_tag})"
   ```
*Status*: Zero mentions of `boundary_status`, `MARGINAL`, `DECISIVE`, or `\pm 0.59\text{ m}` error budget. The broadcast visual gives 100% certainty to sub-centimeter margin calls.

---

### E. `benchmark_end_to_end_eval_001.py` (Lines 231-237)
```python
# Under the engineering review policy:
# |Delta X| > 0.59m -> DECISIVE
# |Delta X| <= 0.59m -> MARGINAL
# Invalid geometry / occluded contact -> UNDETERMINED
if res.decision == "UNDETERMINED":
    pred_policy = "UNDETERMINED"
elif abs(res.margin_val) <= empirical_p95_m:
    pred_policy = "MARGINAL"
else:
    pred_policy = "DECISIVE"
```
*Status*: The benchmark test harness **independently computed** `pred_policy` on top of `res.margin_val`, bypassing the engine's internal `res.boundary_status`. This explains why the benchmark reported 96% policy accuracy while the live application lacked the policy entirely!

---

## 3. Implementation Gap Matrix

| Architectural Dimension | Current Implementation | Expected Implementation | Gap Severity |
|---|---|---|---|
| **Calibration Error Budget** | Hardcoded `0.27 m` (holdout on `0.jpg`) | `0.59 m` ($P_{95}$ from `GEOMETRY-EVAL-001`) | **High** (Uncertainty threshold 54% too narrow) |
| **Computational Level Tolerance** | `LEVEL_TOLERANCE_M = 0.05` | `LEVEL_TOLERANCE_M = 0.05` | None (Preserved as numerical zero guard) |
| **`EvidenceSnapshot` Data Contract** | Drops `boundary_status` and `geometry_status` | Must store `boundary_status` and `error_budget_m` | **High** (Data dropped across component boundary) |
| **State Machine Awareness** | Freezes identically for marginal & decisive offside | Distinct state / flag for `MARGINAL_REVIEW` | **Medium** (Policy awareness missing in engine) |
| **Broadcast VAR Banner** | `DECISION NO GOAL - OFFSIDE` on $+0.06\text{ m}$ | `VAR: OFFSIDE (MARGINAL REVIEW RECOMMENDED)` | **Critical** (Misrepresents sub-budget call as absolute fact) |
| **Telemetry Sub-Pill** | `MARGIN: +0.06m` | `MARGIN: +0.06m [MARGINAL ±0.59m]` | **Critical** (Hides calibration uncertainty from operator) |

---

## 4. Specific Remediation Plan

To formally bridge this gap without breaking backward compatibility:

1. **Update `src/engine/offside_engine.py`**:
   - Set `DEFAULT_EMPIRICAL_ERROR_BUDGET_M = 0.59` (derived from `GEOMETRY-EVAL-001` $P_{95} = 0.587\text{ m}$).
   - Allow runtime override via constructor: `OffsideEngine(empirical_error_budget_m=0.59)`.

2. **Update `src/engine/evidence_snapshot.py`**:
   - Add `boundary_status: str = "DECISIVE"` (`"DECISIVE"`, `"MARGINAL"`, `"UNDETERMINED"`).
   - Add `empirical_error_budget_m: float = 0.59`.
   - Add `geometry_status: str = "NORMAL"` (`"NORMAL"`, `"SUSPECT"`, `"INVALID"`).

3. **Update `src/engine/play_state.py`**:
   - Pass `boundary_status`, `empirical_error_budget_m`, and `geometry_status` into `EvidenceSnapshot` from `last_offside_result`.

4. **Update `app.py` UI & Overlay**:
   - If `boundary_status == "MARGINAL"`:
     - Top TV Banner: `"VAR: OFFSIDE POSITION (MARGINAL — ON-FIELD CALL STANDS / REVIEW)"`
     - Sub-pill: `f"MARGIN: {sign}{margin_val:.2f}m [MARGINAL ±{budget:.2f}m] | STATUS: WITHIN CALIBRATION UNCERTAINTY"`
   - If `boundary_status == "DECISIVE"`:
     - Top TV Banner: `"VAR: OFFSIDE POSITION DETECTED (DECISIVE)"`
     - Sub-pill: `f"MARGIN: {sign}{margin_val:.2f}m [DECISIVE] | ATTACK: TEAM {attack_team_id}"`

5. **Benchmark Correction (E17)**:
   - In `benchmark_end_to_end_eval_001.py`, update Event E17 ground truth policy from `"DECISIVE"` to `"MARGINAL"` since its ground truth margin is $0.00\text{ m} \le 0.59\text{ m}$.
