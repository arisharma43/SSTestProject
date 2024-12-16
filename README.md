# Video Arithmetic Solver

This repository implements a solution to accurately compute the result of an arithmetic sequence displayed in a video. Two distinct approaches are utilized to achieve this goal:

1. **Approach 1**: Traditional Image Processing Techniques (OpenCV/skimage)
2. **Approach 2**: Deep Learning-based Solution using LeNet (PyTorch)

## Methodologies

### Approach 1: Image Processing Techniques
- **Tools Used**: OpenCV
- **Description**:
  - Thresholding to isolate digits and operators.
  - Contour detection to extract regions of interest (ROIs).
  - Edge detection and bounding box calculations to locate characters.
  - Character recognition based on handcrafted rules.
- **Challenges Addressed**:
  - Noise filtering to improve accuracy.
  - Precise extraction of symbols from video frames.

### Approach 2: Deep Learning with LeNet
- **Tools Used**: PyTorch, OpenCV
- **Description**:
  - A modified LeNet model trained on a custom dataset combining MNIST digits and operator images.
  - Preprocessing steps include resizing images to 28x28 and normalizing pixel values.
  - The trained model identifies digits and operators from video frames.
- **Advantages**:
  - High accuracy with minimal manual preprocessing.
  - Ability to generalize across varied video inputs.

## Repository Structure
- `data/`: Contains datasets (MNIST, operator images) and sample videos.
- `models/`: Pretrained models and saved checkpoints.
- `src/`: Implementation of both approaches.
  - `image_processing.py`: Approach 1 (traditional techniques).
  - `lenet_model.py`: Approach 2 (deep learning-based solution).
- `README.md`: Documentation.

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (for training and inference with PyTorch)
- Libraries:
  - OpenCV
  - PyTorch
  - NumPy

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/video-arithmetic-solver.git
   cd video-arithmetic-solver
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Approach 1: Image Processing
1. Navigate to the `src/` directory.
   ```bash
   cd src
   ```
2. Run the script for Approach 1:
   ```bash
   python image_processing.py --video_path <path_to_video>
   ```
   Replace `<path_to_video>` with the path to your input video file.

### Approach 2: Deep Learning
1. Train the LeNet model (optional, pretrained model is included):
   ```bash
   python lenet_model.py --train --data_dir <path_to_data>
   ```
   Replace `<path_to_data>` with the directory containing MNIST and operator images.

2. Process a video:
   ```bash
   python lenet_model.py --video_path <path_to_video>