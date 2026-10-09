"""
Second-Last Opponent Selector implementing FIFA Law 11.
Rules:
1. Referees (class_id == 2 or role == 'referee') are strictly excluded.
2. Defending team players are identified.
3. Goalkeepers (class_id == 1) are treated as regular opponents and NOT hardcoded.
4. Opponents are sorted by physical distance to their own defending goal line:
   - For attacking_direction_sign == +1: Defending goal is at X = 105m (opponents defending right).
   - For attacking_direction_sign == -1: Defending goal is at X = 0m (opponents defending left).
5. The second-last opponent is dynamically selected.
"""

from typing import List, Tuple, Optional, Any, Dict
import numpy as np


class DefenderSelector:
    @staticmethod
    def select_second_last_opponent(
        players: List[Dict[str, Any]],
        defend_team_id: int = 1,
        attack_direction: str = 'right',
        pitch_geom: Optional[Any] = None,
        attacking_direction_sign: int = 1
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], float]:
        """
        Dynamically isolates the defending team and identifies the second-last opponent.
        Returns:
          (last_opponent, second_last_opponent, baseline_coord)
        """
        # Rule 1: Exclude referees
        non_referees = [
            p for p in players
            if p.get("class_id") != 2 and p.get("role") != "referee"
        ]

        # Rule 2: Filter defending team players
        # Strictly require defender team match, or defending goalkeeper (class_id 1 not on attack team)
        attack_team_id = 1 if defend_team_id == 0 else 0
        defenders = []
        for p in non_referees:
            c_id = p.get("class_id", 0)
            t_id = p.get("team_id")
            if t_id == defend_team_id:
                defenders.append(p)
            elif c_id == 1 and t_id != attack_team_id:
                defenders.append(p)

        # Fallback only if no defenders identified at all: use non-referees not on attack team
        if not defenders:
            defenders = [p for p in non_referees if p.get("team_id") != attack_team_id]

        if not defenders:
            return None, None, 0.0

        # Rule 3 & 4: Sort defenders by physical distance to their defending goal line
        # If calibrated metric pitch homography is available:
        if pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
            try:
                pitch_length = getattr(pitch_geom, "pitch_length", 105.0)

                # Check if all defenders already have world_x specified
                has_all_world = all("world_x" in p for p in defenders)
                if not has_all_world:
                    feet_px = []
                    for p in defenders:
                        if "ground_point" in p:
                            feet_px.append([p["ground_point"][0], p["ground_point"][1]])
                        elif "bbox" in p:
                            cx = (p["bbox"][0] + p["bbox"][2]) / 2.0
                            feet_px.append([cx, float(p["bbox"][3])])
                        else:
                            feet_px.append([0.0, 0.0])

                    feet_px = np.array(feet_px, dtype=np.float32)
                    pitch_coords = pitch_geom.image_to_pitch(feet_px)

                    for i, p in enumerate(defenders):
                        if "world_x" not in p:
                            p["world_x"] = float(pitch_coords[i, 0])
                            p["world_y"] = float(pitch_coords[i, 1])

                defenders_with_dist = []
                for p in defenders:
                    px = float(p["world_x"])
                    if attacking_direction_sign == 1:
                        # Opponents defending goal at X = pitch_length (105m)
                        dist_to_goal = abs(pitch_length - px)
                    else:
                        # Opponents defending goal at X = 0.0m
                        dist_to_goal = abs(px - 0.0)

                    defenders_with_dist.append((p, dist_to_goal))

                # Sort defenders ascending by distance to their goal line
                defenders_sorted = [x[0] for x in sorted(defenders_with_dist, key=lambda x: x[1])]

                # In football, the goalkeeper is the primary goal guardian stationed at the goal line.
                # Check whether a DEFENDING goalkeeper is visible within the camera frame:
                # 1. Explicit goalkeeper detection belonging to the defending team/half
                # 2. Or a defender positioned directly inside the goalmouth (< 6.0m from defending goal line)
                has_defending_gk = False
                for p in defenders:
                    if p.get("class_id") == 1 or p.get("role") == "goalkeeper":
                        dist_g = abs(pitch_length - float(p["world_x"])) if attacking_direction_sign == 1 else abs(float(p["world_x"]) - 0.0)
                        if dist_g < pitch_length / 2.0:
                            has_defending_gk = True
                            break

                if not has_defending_gk and len(defenders_with_dist) > 0:
                    closest_dist = min(x[1] for x in defenders_with_dist)
                    if closest_dist < 6.0:
                        has_defending_gk = True

                if has_defending_gk:
                    # Defending goalkeeper is visible in-frame: index 0 is GK, index 1 is 2nd-last opponent
                    last_opp = defenders_sorted[0]
                    second_last_opp = defenders_sorted[1] if len(defenders_sorted) >= 2 else defenders_sorted[0]
                else:
                    # Goalkeeper is off-screen at the defending goal line (1st opponent).
                    # Therefore, the deepest visible outfield defender is the 2nd-last opponent!
                    last_opp = None
                    second_last_opp = defenders_sorted[0]

                baseline = float(second_last_opp.get("world_x", 0.0))
                return last_opp, second_last_opp, baseline
            except Exception:
                pass

        # Camera-space sorting fallback (uncalibrated)
        if attack_direction in ['top', 'up']:
            defenders_sorted = sorted(defenders, key=lambda p: float(p.get("bbox", [0, 0, 0, 0])[3]))
        elif attack_direction in ['bottom', 'down']:
            defenders_sorted = sorted(defenders, key=lambda p: float(p.get("bbox", [0, 0, 0, 0])[3]), reverse=True)
        elif attack_direction in ['left', '-X', '-x']:
            defenders_sorted = sorted(defenders, key=lambda p: (p.get("bbox", [0, 0, 0, 0])[0] + p.get("bbox", [0, 0, 0, 0])[2]) / 2.0)
        else:
            # default 'right' / '+X'
            defenders_sorted = sorted(defenders, key=lambda p: (p.get("bbox", [0, 0, 0, 0])[0] + p.get("bbox", [0, 0, 0, 0])[2]) / 2.0, reverse=True)

        last_opp = defenders_sorted[0]
        second_last_opp = defenders_sorted[1] if len(defenders_sorted) >= 2 else defenders_sorted[0]

        if attack_direction in ['top', 'up']:
            baseline_coord = float(second_last_opp.get("bbox", [0, 0, 0, 0])[1])
        elif attack_direction in ['bottom', 'down']:
            baseline_coord = float(second_last_opp.get("bbox", [0, 0, 0, 0])[3])
        elif attack_direction in ['left', '-X', '-x']:
            baseline_coord = float(second_last_opp.get("bbox", [0, 0, 0, 0])[0])
        else:
            baseline_coord = float(second_last_opp.get("bbox", [0, 0, 0, 0])[2])

        return last_opp, second_last_opp, baseline_coord
