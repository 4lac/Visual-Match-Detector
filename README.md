## 🖼️ Visual Match Detector
<p align="center">
  <img src="logo.png" alt="Visual Match Detector Logo" width="320"/>
</p>
<p align="center"><b>Part of <a href="https://github.com/4lac">❄️ R/ICE Tools</a></b> — a collection of security tools by Reyof</p>

A Python desktop tool that compares a face in an **older photo** with a face in a **recent photo** and estimates whether they belong to the same person.

### How it works
1. **Face detection** — finds faces in each photo using YuNet (OpenCV DNN).
2. **Alignment** — straightens and crops the largest face.
3. **Face fingerprint** — converts the face into a 128-number embedding using SFace.
4. **Comparison** — measures how close the two fingerprints are (cosine similarity).

### Verdicts
| Similarity | Verdict |
|---|---|
| ≥ 0.50 | 🟢 Likely the same person |
| 0.363 – 0.50 | 🟡 Possible match — needs human review |
| < 0.363 | 🔴 Likely different people |

> The tool never makes a final identification on its own. Borderline results are flagged for a human to review, because face recognition can produce false matches.

### Features
- Load two photos and see the detected face highlighted
- Similarity score with a color-coded verdict
- Handles photos with several faces (uses the largest one)
- Works with non-English file paths
- Desktop app and command-line mode

### Installation
```bash
pip install -r requirements.txt
```
The two face models (~38 MB) are downloaded automatically from the official OpenCV model zoo on first run.

### Usage
**Desktop app**
```bash
python app.py
```
**Command line**
```bash
python app.py old_photo.jpg new_photo.jpg
```

### Roadmap
- **Phase 1 — Two-photo comparison** ✅
- **Phase 2 — Accuracy evaluation** on a cross-age research dataset (CACD / AgeDB), with measured accuracy and error rates
- **Phase 3 — Video input** — compare faces from a video against a reference list, producing a ranked list for human review

### Technologies Used
- Python
- OpenCV (YuNet face detector, SFace face recognizer)
- Tkinter
- NumPy
- Pillow

### Responsible Use
This project is for **educational and research purposes**. Facial data is sensitive personal data — only use photos you have the right to process, and follow local data-protection laws (e.g. Saudi PDPL).
