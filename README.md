# Image Zooming

This project implements various image zooming (resizing) techniques using Python and OpenCV, including Nearest Neighbor, Bilinear, and Bicubic interpolation methods. It also evaluates and compares the quality of the zoomed images using PSNR (Peak Signal-to-Noise Ratio).

## 🔍 Features

- Image zooming with:
  - Nearest Neighbor Interpolation
  - Bilinear Interpolation
  - Bicubic Interpolation
- Quality comparison using PSNR
- Side-by-side visual comparison of zoomed images
- Easy-to-use script structure

## 📁 Project Structure
Image_Zooming/
│
├── original_image.jpg # Sample input image
├── zooming.py # Main script to perform zooming
├── utils.py # Utility functions for interpolation and PSNR
├── results/ # Folder to save output images
└── README.md # Project documentation

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt


# Run the script
python zooming.py

