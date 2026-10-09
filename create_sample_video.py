"""
Create a sample match video clip from the sequential dataset images.
Resizes frames consistently to standard Full HD (1920x1080).
"""

import os
import cv2


def make_sample_video(output_path: str = "sample_match.mp4", max_frames: int = 60, fps: int = 15):
    img_dir = r"e:\ODS\Offside_Images"
    files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 9999
    )[:max_frames]

    if not files:
        print("No images found in", img_dir)
        return

    target_w, target_h = 1920, 1080

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    print(f"Creating sample video '{output_path}' ({target_w}x{target_h}, {len(files)} frames @ {fps} FPS)...")
    for f in files:
        img_p = os.path.join(img_dir, f)
        img = cv2.imread(img_p)
        if img is not None:
            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            writer.write(img)

    writer.release()
    print(f"Sample video created successfully: {output_path}")


if __name__ == "__main__":
    make_sample_video()
