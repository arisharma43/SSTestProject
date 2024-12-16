import cv2
import numpy as np
import os


def load_templates(template_dir):
    templates = {}
    operator_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "plus": 10,
        "minus": 11,
        "times": 12,
        "div": 13,
    }
    for file in os.listdir(template_dir):
        if file.endswith(".png"):
            key = file.split(".")[0]
            if key in operator_map:
                img = cv2.imread(os.path.join(template_dir, file), cv2.IMREAD_GRAYSCALE)
                templates[operator_map[key]] = img
    return templates


def match_template(roi, templates):
    best_match = None
    best_score = float("inf")
    for label, template in templates.items():
        template_resized = cv2.resize(template, roi.shape[::-1])
        result = cv2.matchTemplate(roi, template_resized, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(result)
        if min_val < best_score:
            best_score = min_val
            best_match = label
    return best_match


def process_video_image_processing(video_path, template_dir):
    templates = load_templates(template_dir)
    cap = cv2.VideoCapture(video_path)
    recognized_sequence = []
    last_prediction = None

    # Process every 5th frame to avoid duplicate frames
    frame_skip = 5

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold and find contours
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter noise
            if w > 10 and h > 10:
                roi = gray[y : y + h, x : x + w]

                # Recognize digit/operator
                predicted = match_template(roi, templates)

                # Avoid duplicates
                if predicted is not None and predicted != last_prediction:
                    recognized_sequence.append(predicted)
                    last_prediction = predicted

    cap.release()

    operator_map = {10: "+", 11: "-", 12: "*", 13: "/"}
    expression = "".join(operator_map.get(x, str(x)) for x in recognized_sequence)
    print(f"Recognized Expression: {expression}")
    try:
        result = eval(expression)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error in evaluating expression: {e}")


# modify path to correct video and images paths
VIDEO_PATH = "D:/Coding/ResearchAssignment/SSTestProject/data/video/video_2.mp4"
TEMPLATE_DIR = "D:/Coding/ResearchAssignment/SSTestProject/data/images"
process_video_image_processing(VIDEO_PATH, TEMPLATE_DIR)
