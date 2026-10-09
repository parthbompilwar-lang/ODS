"""
Quick test script to run the Offside Detection Pipeline on sample dataset images.
"""

import os
import cv2
import json
from src.pipeline import OffsideDetectorPipeline


def main():
    image_dir = r"e:\ODS\Offside_Images"
    output_dir = r"e:\ODS\demo_results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize end-to-end pipeline
    pipeline = OffsideDetectorPipeline(
        pose_model="yolo11n-pose.pt",
        det_model="yolo11n.pt",
        device="cuda"
    )

    # Select sample images from dataset
    sample_images = ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg"]

    for img_name in sample_images:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        print(f"\nProcessing {img_name}...")
        image = cv2.imread(img_path)
        if image is None:
            continue

        annotated_frame, result = pipeline.process_frame(
            image=image,
            attack_team_id=0,
            attack_direction='right',
            draw_radar=True,
            draw_skeletons=True
        )

        out_path = os.path.join(output_dir, f"var_result_{img_name}")
        cv2.imwrite(out_path, annotated_frame)
        print(f"Decision: {result.decision} | Margin: {result.margin_val}{result.margin_unit}")
        print(f"Explanation: {result.explanation}")
        print(f"Saved VAR broadcast graphic to: {out_path}")


if __name__ == "__main__":
    main()
