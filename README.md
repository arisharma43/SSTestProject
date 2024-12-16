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
- `approach_1/`: Approach 1 (traditional techniques)
  - `main.py`: Primary code for approach 1
- `approach_2/`: Approach 2 (LeNet solution)
  - `lenet_model.pth`: Saved model
  - `main.py`: Primary code for approach 2
- `data/`: Contains datasets (MNIST, operator images) and sample videos.
  -`images/`: Dataset of images
  -`MNIST_dataset`: MNIST dataset
  -`video`: 2 videos to use
- `README.md`: Documentation.

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- Libraries:
  - OpenCV
  - PyTorch
  - NumPy

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/arisharma43/SSTestProject.git
   cd SSTestProject
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
1. Navigate to the `approach_1/` directory.
   ```bash
   cd approach_1
   ```
2. Run the script for Approach 1:
   Make sure to modify the VIDEO_PATH and TEMPLATE_DIR paths (lines 102,103) to be the exact paths on your system
   
   ```bash
   python main.py
   ```

### Approach 2: Deep Learning
1. Navigate to the `approach_2/` directory.
   ```bash
   cd approach_2
   ```

2. Run the script for Approach 2:
   Make sure to modify the MNIST_PATH, TEMPLATE_DIR, and VIDEO_PATH paths (lines 92,97,232) to be the exact paths on your system
   
   ```bash
   python main.py
   ```
