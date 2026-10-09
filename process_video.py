"""
Automated Video Offside Detection & Interactive VAR Review System.
Processes match videos frame-by-frame, performs FIFA Law 11 offside analysis,
and renders Premier League broadcast-grade VAR graphics.

Usage:
  py -3.11 process_video.py --video sample_match.mp4 --output var_output.mp4 --direction left --team 0
  py -3.11 process_video.py --video sample_match.mp4 --display
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from tqdm import tqdm

from src.pipeline import OffsideDetectorPipeline
from src.geometry.pitch_geometry import PitchGeometry
from src.engine.offside_engine import OffsideResult


class VideoOffsideProcessor:
    def __init__(
        self,
        model_path: str = "models/yolo11_offside_best.pt",
        device: str = "cuda"
    ):
        print(f"Loading Offside Detection Pipeline with model: {model_path}...")
        self.pipeline = OffsideDetectorPipeline(
            pose_model="yolo11n-pose.pt",
            det_model=model_path if os.path.exists(model_path) else "yolo11n.pt",
            device=device
        )
        # Temporal smoothing for vanishing point
        self.smoothed_vp = None
        self.vp_alpha = 0.25  # Exponential moving average weight

    def process_video(
        self,
        video_path: str,
        output_path: str = "var_analyzed_match.mp4",
        attack_direction: str = "left",
        attack_team_id: int = 0,
        display: bool = False,
        interactive: bool = False,
        max_frames: int = None,
        draw_radar: bool = True,
        draw_skeletons: bool = True
    ):
        if not os.path.exists(video_path):
            print(f"Error: Input video '{video_path}' does not exist.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file '{video_path}'.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("=" * 65)
        print("PREMIER LEAGUE VAR VIDEO OFFSIDE DETECTOR")
        print("=" * 65)
        print(f"Input Video:        {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
        print(f"Attacking Direction:{attack_direction.upper()} | Attacking Team: {attack_team_id}")
        print(f"Saving VAR Output:  {output_path}")
        if display:
            print("Display Mode:       ENABLED (Interactive window)")
            print("Controls: [SPACE]=Pause/Play  [D/A]=Step Frame  [T]=Swap Team  [G]=Swap Goal  [S]=Save  [Q]=Exit")
        print("=" * 65)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        pbar = tqdm(total=total_frames, desc="Processing Video Frames", unit="frame")

        frame_idx = 0
        paused = interactive  # Start paused if in interactive review mode

        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                curr_frame = frame.copy()
            else:
                # In paused mode, reuse current frame
                frame = curr_frame.copy()

            # 1. Pitch vanishing point with temporal smoothing across video
            raw_vp = self.pipeline.pitch_geom.estimate_vanishing_point(frame, goal_direction=attack_direction)
            if self.smoothed_vp is None:
                self.smoothed_vp = raw_vp
            else:
                self.smoothed_vp = (
                    self.vp_alpha * raw_vp[0] + (1 - self.vp_alpha) * self.smoothed_vp[0],
                    self.vp_alpha * raw_vp[1] + (1 - self.vp_alpha) * self.smoothed_vp[1]
                )

            # 2. Run offside analysis
            annotated, result = self.pipeline.process_frame(
                image=frame,
                attack_team_id=attack_team_id,
                attack_direction=attack_direction,
                manual_vp=self.smoothed_vp,
                draw_radar=draw_radar,
                draw_skeletons=draw_skeletons
            )

            # Add frame counter overlay in bottom left corner
            cv2.putText(
                annotated,
                f"FRAME {frame_idx}/{total_frames} | {result.decision} ({result.margin_val}{result.margin_unit})",
                (30, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if not paused:
                writer.write(annotated)
                pbar.update(1)

            # Interactive / Display Controls
            if display:
                disp_w = min(1280, width)
                disp_h = int(height * (disp_w / width))
                display_frame = cv2.resize(annotated, (disp_w, disp_h))

                win_title = f"VAR Offside Review Studio - Frame {frame_idx} [{'PAUSED' if paused else 'PLAYING'}]"
                cv2.imshow("VAR Offside Review Studio", display_frame)

                delay = 0 if paused else 1
                key = cv2.waitKey(delay) & 0xFF

                if key in [ord('q'), 27]:  # Q or ESC to exit
                    print("\nProcessing interrupted by user.")
                    break
                elif key == 32:  # Space to toggle pause
                    paused = not paused
                elif key in [ord('d'), 83]:  # D or Right Arrow to advance frame
                    paused = True
                    ret, frame = cap.read()
                    if ret:
                        frame_idx += 1
                        curr_frame = frame.copy()
                        writer.write(annotated)
                        pbar.update(1)
                elif key == ord('t'):  # T to switch attacking team
                    attack_team_id = 1 if attack_team_id == 0 else 0
                    print(f"\nSwitched Attacking Team to: Team {attack_team_id}")
                elif key == ord('g'):  # G to switch attacking direction
                    attack_direction = "right" if attack_direction == "left" else "left"
                    self.smoothed_vp = None
                    print(f"\nSwitched Attacking Direction to: {attack_direction.upper()}")
                elif key == ord('s'):  # S to save snapshot
                    snap_name = f"var_snapshot_frame_{frame_idx}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"\nSaved VAR snapshot: {snap_name}")

        pbar.close()
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

        print("\n" + "=" * 65)
        print("VIDEO PROCESSING COMPLETE")
        print("=" * 65)
        print(f"Output Video Saved: {os.path.abspath(output_path)}")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Premier League VAR Offside Video Processor")
    parser.add_argument("--video", type=str, default="sample_match.mp4", help="Path to input match video file")
    parser.add_argument("--output", type=str, default="var_output.mp4", help="Path to save output VAR video")
    parser.add_argument("--direction", type=str, default="left", choices=["left", "right"], help="Attacking direction towards opponent goal")
    parser.add_argument("--team", type=int, default=0, choices=[0, 1], help="Attacking team ID (0 or 1)")
    parser.add_argument("--model", type=str, default="models/yolo11_offside_best.pt", help="Path to trained YOLOv11 model weights")
    parser.add_argument("--display", action="store_true", help="Show video playback window with real-time VAR overlays")
    parser.add_argument("--interactive", action="store_true", help="Open interactive VAR review player (pause/step/scrub)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no-radar", action="store_true", help="Disable 2D radar overlay")
    parser.add_argument("--no-skeletons", action="store_true", help="Disable player skeleton overlay")

    args = parser.parse_args()

    processor = VideoOffsideProcessor(model_path=args.model)
    processor.process_video(
        video_path=args.video,
        output_path=args.output,
        attack_direction=args.direction,
        attack_team_id=args.team,
        display=args.display or args.interactive,
        interactive=args.interactive,
        max_frames=args.max_frames,
        draw_radar=not args.no_radar,
        draw_skeletons=not args.no_skeletons
    )


if __name__ == "__main__":
    main()
