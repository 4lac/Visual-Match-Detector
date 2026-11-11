## 🖼️ Visual Match Detector
<p align="center">
  <img src="logo.png" alt="Visual Match Detector Logo" width="320"/>
</p>
<p align="center"><b>Part of <a href="https://github.com/4lac">❄️ R/ICE Tools</a></b> — a collection of security tools by Reyof</p>

A Python desktop tool that compares a face in an **older photo** with a face in a **recent photo** and estimates whether they belong to the same person.

### How it works
1. **Face detection** — finds faces and 5 landmarks (eyes, nose, mouth corners) using YuNet.
2. **Alignment** — uses the landmarks to straighten and crop the largest face.
3. **Face fingerprint** — two AI models turn the face into an embedding:
   - **ArcFace** (InsightFace `w600k_r50`) — the main judge
   - **SFace** (OpenCV) — a second opinion
4. **Comparison** — cosine similarity for each model, then the two opinions are combined.

### Verdicts
ArcFace decides. SFace acts as a safety check: if the two models land on opposite ends (one says *match*, the other says *different*), the result becomes **needs human review**.

| ArcFace similarity | Verdict |
|---|---|
| ≥ 0.45 | 🟢 Likely the same person |
| 0.30 – 0.45 | 🟡 Possible match — needs human review |
| < 0.30 | 🔴 Likely different people |

> ⚠️ These ArcFace thresholds are **provisional**. They will be calibrated in Phase 2 on a real cross-age dataset.
>
> The tool never makes a final identification on its own. Borderline or conflicting results are flagged for a human to review, because face recognition can produce false matches — especially with look-alikes, extreme angles, filters, or low-quality images.

If `onnxruntime` is not installed, the tool falls back to SFace only.

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
The face models are downloaded automatically on first run (~330 MB in total: YuNet and SFace from the OpenCV model zoo, ArcFace from InsightFace).

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
- ONNX Runtime (ArcFace face recognizer)
- Tkinter
- NumPy
- Pillow

### Model Licenses
The ArcFace model (InsightFace `buffalo_l`) is released by InsightFace for **non-commercial research use only**. YuNet and SFace are from the OpenCV model zoo.

### Responsible Use
This project is for **educational and research purposes**. Facial data is sensitive personal data — only use photos you have the right to process, and follow local data-protection laws (e.g. Saudi PDPL).
