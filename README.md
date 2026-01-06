## 🖼️ Visual Match Detector

A lightweight Python desktop application for comparing two images and detecting visual similarities using computer vision techniques.

### Description
The project focuses on comparing two images using computer vision techniques such as ORB feature matching and homography to identify visual similarity. This version represents the core foundation of a larger concept, with future plans to explore security-related use cases built on top of this functionality.

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
