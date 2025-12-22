## 🖼️ Visual Match Detector

A lightweight Python desktop application for comparing two images and detecting visual similarities using computer vision techniques.

### Description
This project allows users to upload two images, preview them, and visually compare them using ORB feature matching and homography detection.

### Features
- Upload and preview two images
- Detect visual matches using ORB
- Highlight matched regions between images
- Simple Tkinter-based GUI

### Known Issues
⚠️ This project is still under development.  
There are known UI limitations, such as image scaling and canvas refresh behavior when loading new images after a comparison.

### Technologies Used
- Python
- OpenCV
- Tkinter
- NumPy
- Pillow

### How to Run
```bash
pip install opencv-python numpy pillow
python main.py
