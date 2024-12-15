# import cv2 as cv
# import numpy as np
# import os

# operator_map = {"plus": "+", "minus": "-", "times": "*", "div": "/"}


# # Load templates for digits and operators
# def preprocess_template(template):
#     _, binary = cv.threshold(template, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     return binary


# # Load templates with preprocessing
# def load_templates(template_dir):
#     templates = {}
#     for filename in os.listdir(template_dir):
#         if filename.endswith(".png"):
#             key = filename.split(".")[0]  # Use filename as key
#             filepath = os.path.join(template_dir, filename)
#             template = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#             if template is not None:
#                 processed_template = preprocess_template(template)
#                 processed_template_resized = cv.resize(processed_template, (50, 50))
#                 templates[key] = processed_template_resized  # Resize here
#             else:
#                 print(f"Failed to load {filepath}")
#     return templates


# # Preprocess the frame
# def preprocess_frame(frame):
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     binary = cv.adaptiveThreshold(
#         gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 3
#     )
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
#     cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
#     return cleaned


# # Extract regions of interest (ROIs) for digits/operators
# # Extract regions of interest (ROIs) for digits/operators
# def extract_rois(binary_frame):
#     contours, _ = cv.findContours(
#         binary_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
#     )
#     rois = []
#     for cnt in contours:
#         x, y, w, h = cv.boundingRect(cnt)
#         # Filter by size (adjust these thresholds as needed)
#         if 20 < w < 600 and 20 < h < 800:  # Ensure reasonable sizes
#             roi = binary_frame[y : y + h, x : x + w]
#             rois.append(roi)  # Do not include the x-coordinate for sorting
#     return rois


# # Match ROIs to templates
# # def match_templates(rois, templates):
# #     results = []
# #     for roi in rois:
# #         roi_resized = cv.resize(roi, (50, 50))
# #         roi_resized = roi_resized.astype(np.float32) / 255.0  # Normalize ROI
# #         best_match = None
# #         best_score = -1
# #         for key, template in templates.items():
# #             template_norm = template.astype(np.float32) / 255.0
# #             match = cv.matchTemplate(roi_resized, template_norm, cv.TM_CCOEFF_NORMED)
# #             _, score, _, _ = cv.minMaxLoc(match)
# #             if score > best_score:
# #                 best_score = score
# #                 best_match = key
# #         if best_match in operator_map:
# #             results.append(operator_map[best_match])
# #         else:
# #             results.append(best_match)
# #     return results
# def match_templates(rois, templates):
#     """Match each ROI to the closest template using normalized cross-correlation."""
#     results = []
#     for roi in rois:
#         roi_resized = cv.resize(roi, (50, 50))
#         roi_resized = roi_resized.astype(np.float32) / 255.0  # Normalize ROI
#         best_match = None
#         best_score = -1
#         for key, template in templates.items():
#             template_resized = cv.resize(template, (50, 50))
#             template_norm = template_resized.astype(np.float32) / 255.0
#             match = cv.matchTemplate(roi_resized, template_norm, cv.TM_CCOEFF_NORMED)
#             _, score, _, _ = cv.minMaxLoc(match)
#             if score > best_score:
#                 best_score = score
#                 best_match = key
#         # Map operator names to symbols if applicable
#         if best_match in operator_map:
#             results.append(operator_map[best_match])
#         else:
#             results.append(best_match)
#     return results


# # Compute the result of the arithmetic sequence
# def compute_result(sequence):
#     expression = "".join(sequence)
#     print(f"Recognized Expression: {expression}")
#     try:
#         result = eval(expression)
#     except Exception as e:
#         print(f"Error computing result: {e}")
#         result = None
#     return result


# # Main processing function
# def process_video(video_path, template_dir):
#     """Process video frame by frame to recognize and compute results."""
#     templates = load_templates(template_dir)
#     print("Templates loaded successfully.")

#     cap = cv.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         binary_frame = preprocess_frame(frame)
#         rois = extract_rois(binary_frame)
#         if not rois:
#             continue  # Skip frames with no ROIs

#         sequence = match_templates(rois, templates)
#         if sequence:
#             print(f"Recognized Sequence: {sequence}")
#             result = compute_result(sequence)
#             print(f"Result: {result}")

#     cap.release()
#     print("Processing complete.")


# # Example usage
# if __name__ == "__main__":
#     TEMPLATE_DIR = "D:/Coding/ResearchAssignment/data/data/images"
#     VIDEO_PATH = "D:/Coding/ResearchAssignment/data/data/video/video_1.mp4"
#     process_video(VIDEO_PATH, TEMPLATE_DIR)
import cv2 as cv
import numpy as np
import os

operator_map = {"plus": "+", "minus": "-", "times": "*", "div": "/"}


# Load templates for digits and operators
def preprocess_template(template):
    _, binary = cv.threshold(template, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary


# Load templates with preprocessing
def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            key = filename.split(".")[0]  # Use filename as key
            filepath = os.path.join(template_dir, filename)
            template = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            if template is not None:
                processed_template = preprocess_template(template)
                templates[key] = cv.resize(processed_template, (50, 50))
            else:
                print(f"Failed to load {filepath}")
    return templates


# Preprocess the frame
def preprocess_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 3
    )
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return cleaned


# Extract regions of interest (ROIs) for digits/operators
def extract_rois(binary_frame):
    contours, _ = cv.findContours(
        binary_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    rois = []
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if 20 < w < 600 and 20 < h < 800:
            roi = binary_frame[y : y + h, x : x + w]
            rois.append((x, roi, (x, y, w, h)))  # Include bounding box info
    rois = sorted(rois, key=lambda r: r[0])  # Sort left-to-right
    return rois


# Match ROIs to templates
def match_templates(rois, templates):
    results = []
    labels = []  # For debugging
    for roi_info in rois:
        _, roi, bbox = roi_info
        roi_resized = cv.resize(roi, (50, 50))
        roi_resized = roi_resized.astype(np.float32) / 255.0  # Normalize ROI
        best_match = None
        best_score = -1
        for key, template in templates.items():
            template_norm = template.astype(np.float32) / 255.0
            match = cv.matchTemplate(roi_resized, template_norm, cv.TM_CCOEFF_NORMED)
            _, score, _, _ = cv.minMaxLoc(match)
            if score > best_score:
                best_score = score
                best_match = key
        # Map operator names to symbols
        if best_match in operator_map:
            results.append(operator_map[best_match])
        else:
            results.append(best_match)
        labels.append((bbox, results[-1]))  # Append bounding box and label
    return results, labels


# Compute the result of the arithmetic sequence
def compute_result(sequence):
    expression = "".join(sequence)
    print(f"Recognized Expression: {expression}")
    try:
        result = eval(expression)
    except Exception as e:
        print(f"Error computing result: {e}")
        result = None
    return result


# Main processing function
def process_video(video_path, template_dir):
    templates = load_templates(template_dir)
    print("Templates loaded successfully.")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        binary_frame = preprocess_frame(frame)
        rois = extract_rois(binary_frame)
        if not rois:
            continue

        sequence, labels = match_templates(rois, templates)
        result = compute_result(sequence)

        # Debug visualization
        debug_frame = frame.copy()
        for bbox, label in labels:
            x, y, w, h = bbox
            cv.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(
                debug_frame,
                label,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv.imshow("Frame with Debugging", debug_frame)
        cv.imshow("Binary Frame", binary_frame)
        print(f"Recognized Sequence: {sequence}, Result: {result}")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    print("Processing complete.")


# Example usage
if __name__ == "__main__":
    TEMPLATE_DIR = "D:/Coding/ResearchAssignment/data/data/images"
    VIDEO_PATH = "D:/Coding/ResearchAssignment/data/data/video/video_1.mp4"
    process_video(VIDEO_PATH, TEMPLATE_DIR)
