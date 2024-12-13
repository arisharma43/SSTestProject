import cv2 as cv
import numpy as np
import os

operator_map = {"plus": "+", "minus": "-", "times": "*", "divided_by": "/"}


# Load templates for digits and operators
def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            print("Loading template")
            key = filename.split(".")[0]  # Use filename (e.g., '0', '+') as key

            filepath = os.path.join(template_dir, filename)
            template = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Failed to load {filepath}")
            else:
                print(f"Loaded template: {filename}, Shape: {template.shape}")
                templates[key] = template
    return templates


# Preprocess the frame
def preprocess_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return cleaned


# def preprocess_frame(frame):
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     binary = cv.adaptiveThreshold(
#         gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
#     )
#     return binary


# Extract regions of interest (ROIs) for digits/operators
def extract_rois(binary_frame):
    contours, _ = cv.findContours(
        binary_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    rois = []

    # for cnt in contours:
    #     x, y, w, h = cv.boundingRect(cnt)
    #     cv.rectangle(binary_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv.imshow("Contours", binary_frame)
    # cv.waitKey(0)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        # if 10 < w < 100 and 10 < h < 100:  # Filter by size
        roi = binary_frame[y : y + h, x : x + w]
        rois.append((x, roi))  # Include x-coordinate for sorting
    rois.sort(key=lambda x: x[0])  # Sort ROIs by x-coordinate
    return [roi for _, roi in rois]


# Match ROIs to templates
def match_templates(rois, templates):
    results = []
    for roi in rois:
        roi_resized = cv.resize(roi, (28, 28))
        best_match = None
        best_score = float("inf")
        for key, template in templates.items():
            template_resized = cv.resize(template, (28, 28))
            score = cv.norm(roi_resized, template_resized, cv.NORM_L2)
            if score < best_score:
                best_score = score
                best_match = key
        # results.append(best_match)

        if best_match in operator_map:
            results.append(operator_map[best_match])  # Map operator names to symbols
        else:
            results.append(best_match)  # Keep digits as they are
    # print(results)
    return results


# Compute the result of the arithmetic sequence
def compute_result(sequence):
    expression = "".join(sequence)
    try:
        result = eval(expression)
    except Exception as e:
        print(f"Error computing result: {e}")
        result = None
    return result


# Main processing function
def process_video(video_path, template_dir):
    templates = load_templates(template_dir)
    print("Processing video")
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        binary_frame = preprocess_frame(frame)
        rois = extract_rois(binary_frame)
        sequence = match_templates(rois, templates)

        if sequence:
            result = compute_result(sequence)
            print(f"Sequence: {sequence}, Result: {result}")

    cap.release()


# Example usage
if __name__ == "__main__":
    TEMPLATE_DIR = "D:/Coding/ResearchAssignment/data/data/images"
    VIDEO_PATH = "D:/Coding/ResearchAssignment/data/data/video/video_1.mp4"
    process_video(VIDEO_PATH, TEMPLATE_DIR)
