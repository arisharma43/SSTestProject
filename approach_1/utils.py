import cv2
import numpy as np
import os


def load_templates(template_dir):
    """Load all templates from a directory."""
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            filepath = os.path.join(template_dir, filename)
            template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            templates[filename.split(".")[0]] = template
    return templates


def log_roi_sizes(frame, templates, threshold=0.8):
    """Perform template matching and log ROI sizes."""
    roi_sizes = []
    for name, template in templates.items():
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            w, h = template.shape[::-1]
            roi_sizes.append((w, h, name))
    return roi_sizes


def analyze_roi_sizes(video_path, template_dir, sample_frames=10, threshold=0.8):
    """Analyze ROI sizes for a set of frames in a video."""
    # Load templates
    templates = load_templates(template_dir)

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)

    all_roi_sizes = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame at index {idx}.")
            continue

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Log ROI sizes for this frame
        roi_sizes = log_roi_sizes(gray_frame, templates, threshold)
        all_roi_sizes.extend(roi_sizes)

    cap.release()

    # Analyze results
    if all_roi_sizes:
        widths = [w for w, h, _ in all_roi_sizes]
        heights = [h for w, h, _ in all_roi_sizes]
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        print(f"Average ROI size: Width = {avg_width:.2f}, Height = {avg_height:.2f}")
        print("Detailed ROI sizes (Width x Height, Template Name):")
        for w, h, name in all_roi_sizes:
            print(f"{w} x {h}, {name}")
    else:
        print("No matches found in the analyzed frames.")


if __name__ == "__main__":
    # Path to video file
    VIDEO_PATH = "D:/Coding/ResearchAssignment/data/data/video/video_1.mp4"

    # Path to directory containing templates
    # template_dir = "templates/"  # Update with your template directory
    TEMPLATE_DIR = "D:/Coding/ResearchAssignment/data/data/images"

    # Analyze ROI sizes
    analyze_roi_sizes(VIDEO_PATH, TEMPLATE_DIR)
