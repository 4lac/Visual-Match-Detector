"""
Face verification engine for Visual Match Detector.

Pipeline:
  1. Detect faces + 5 landmarks with YuNet (OpenCV DNN face detector).
  2. Align the largest face using the landmarks.
  3. Turn it into a "face fingerprint" (embedding) with two models:
       - ArcFace (InsightFace w600k_r50) -> main judge, 512 numbers
       - SFace (OpenCV)                  -> second opinion, 128 numbers
  4. Compare fingerprints with cosine similarity and combine the two opinions.

If onnxruntime or the ArcFace model is not available, the engine falls back
to SFace only.
"""

import os
import urllib.request
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
YUNET = ("face_detection_yunet_2023mar.onnx",
         _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = ("face_recognition_sface_2021dec.onnx",
         _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx")
ARCFACE = ("w600k_r50.onnx",
           "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")

# Thresholds: (strong match, possible match) on cosine similarity.
# SFace values come from the SFace authors.
# ArcFace values are PROVISIONAL — to be calibrated in Phase 2 on a real dataset.
THRESHOLDS = {
    "arcface": (0.45, 0.30),
    "sface": (0.50, 0.363),
}

# Where ArcFace expects the 5 landmarks to sit in its 112x112 input
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

MESSAGES = {
    "match": "Likely the same person",
    "review": "Possible match — needs human review",
    "no_match": "Likely different people",
}


@dataclass
class FaceResult:
    box: tuple              # (x, y, w, h) of the chosen face
    faces_found: int        # how many faces were detected in the image
    sface: np.ndarray       # SFace embedding
    arcface: np.ndarray     # ArcFace embedding (None if ArcFace is unavailable)


@dataclass
class Verdict:
    label: str              # final decision: "match", "review", or "no_match"
    message: str
    arcface_score: float    # None if ArcFace is unavailable
    sface_score: float
    models_agree: bool


def _download(url, path):
    print(f"[*] Downloading {os.path.basename(url)} ...")
    urllib.request.urlretrieve(url, path)


def _model_path(name, url):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) < 10_000:
        _download(url, path)
    return path


def _arcface_path():
    """ArcFace ships inside a zip (~288 MB); extract only the model we need."""
    path = os.path.join(MODELS_DIR, ARCFACE[0])
    if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
        return path
    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
    _download(ARCFACE[1], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(ARCFACE[0], MODELS_DIR)
    os.remove(zip_path)
    return path


def _label(score, model):
    strong, possible = THRESHOLDS[model]
    if score >= strong:
        return "match"
    if score >= possible:
        return "review"
    return "no_match"


class FaceEngine:
    def __init__(self, use_arcface=True, det_score_threshold=0.8):
        self.detector = cv2.FaceDetectorYN.create(
            _model_path(*YUNET), "", (320, 320), det_score_threshold, 0.3, 5000)
        self.sface = cv2.FaceRecognizerSF.create(_model_path(*SFACE), "")

        self.arcface = None
        self.arcface_error = None
        if use_arcface:
            try:
                import onnxruntime as ort
                self.arcface = ort.InferenceSession(
                    _arcface_path(), providers=["CPUExecutionProvider"])
                self._arc_input = self.arcface.get_inputs()[0].name
            except Exception as exc:          # missing package, no internet, ...
                self.arcface_error = str(exc)

    # ------------------------------------------------------------ fingerprints
    def _arcface_embedding(self, image, face):
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        aligned = cv2.warpAffine(image, matrix, (112, 112))
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        emb = self.arcface.run(None, {self._arc_input: blob})[0][0]
        return emb / np.linalg.norm(emb)

    def analyze(self, image):
        """Return a FaceResult for the largest face in a BGR image, or None."""
        if image is None:
            raise ValueError("Image could not be read.")
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        face = max(faces, key=lambda f: f[2] * f[3])
        sface_emb = self.sface.feature(self.sface.alignCrop(image, face))
        arc_emb = self._arcface_embedding(image, face) if self.arcface else None
        return FaceResult(box=tuple(int(v) for v in face[:4]), faces_found=len(faces),
                          sface=sface_emb, arcface=arc_emb)

    # ------------------------------------------------------------ comparison
    def compare(self, a, b):
        sface_score = float(self.sface.match(a.sface, b.sface, cv2.FaceRecognizerSF_FR_COSINE))
        sface_label = _label(sface_score, "sface")

        if self.arcface is None:
            return Verdict(sface_label, MESSAGES[sface_label], None, sface_score, True)

        arc_score = float(np.dot(a.arcface, b.arcface))
        arc_label = _label(arc_score, "arcface")

        # ArcFace is the main judge. SFace is a second opinion:
        # if the two models land on opposite ends, nobody gets a confident answer.
        opposite = {arc_label, sface_label} == {"match", "no_match"}
        if opposite:
            final = "review"
            message = "Models disagree — needs human review"
        else:
            final = arc_label
            message = MESSAGES[final]
        return Verdict(final, message, arc_score, sface_score, not opposite)


def load_image(path):
    """cv2.imread that also works with non-ASCII (e.g. Arabic) file paths."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a readable image: {os.path.basename(path)}")
    return image
