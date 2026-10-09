"""
Train YOLOv11 on the Football Offside Detection Dataset.
Compliant with 2025 IEEE research methodology:
Indahsari et al., "Implementation of YOLOv11 for Automatic Offside Detection in Football"
"""

import os
from ultralytics import YOLO
import torch


def train_offside_model(
    epochs: int = 15,
    imgsz: int = 640,
    batch: int = 16,
    model_variant: str = "yolo11n.pt",
    device: str = "0"
):
    print("=" * 65)
    print(f"Starting YOLOv11 Offside Training ({model_variant})")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    yaml_path = os.path.abspath("yolo_offside_dataset/data.yaml")

    # Load pretrained YOLOv11 model
    model = YOLO(model_variant)

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/train",
        name="yolo11_offside",
        plots=True,
        save=True,
        verbose=True
    )

    print("\nTraining complete!")
    best_weights = os.path.join("runs", "train", "yolo11_offside", "weights", "best.pt")
    print(f"Best trained weights saved at: {best_weights}")
    return best_weights


if __name__ == "__main__":
    train_offside_model(epochs=15, batch=16)
