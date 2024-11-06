# Corneal Ulcer Grading with Mask and Classification Models

This project uses a YOLOv8-based mask model and a ResNet50-based classification model to process corneal images. The `main.py` script applies the mask model to isolate the cornea and then classifies the condition's severity.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)

## Prerequisites
- **Python**: Ensure you have a Python version installed that is lower than 3.12. You can check your Python version with:
    ```bash
    python --version
    ```
- **Pipenv**: If `pipenv` is not installed, install it via pip:
    ```bash
    pip install pipenv
    ```

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Set Up the Virtual Environment**:
   Create a `pipenv` environment with Python <3.12 by specifying the version with `--python`:
    ```bash
    pipenv --python 3.11
    ```

3. **Install Dependencies**:
   Install all required libraries from `requirements.txt`:
    ```bash
    pipenv install -r requirements.txt
    ```

## Running the Application

1. **Activate the Environment**:
    ```bash
    pipenv shell
    ```

2. **Run the Main Script**:
    To run the script and process images in a specified folder, use:
    ```bash
    python main.py <path_to_image_folder>
    ```
   Replace `<path_to_image_folder>` with the path to the folder containing the images you want to process.

## Usage
This script processes images in the provided folder, applies a mask model to isolate corneal regions, and then classifies each image based on severity. The predictions are saved as a grid of labeled images in `prediction_grid.jpg`.

### Example
To run the model on images stored in a folder named `test_images`:
```bash
python main.py test_images
