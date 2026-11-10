"""
Face verification engine for Visual Match Detector.

Pipeline:
  1. Detect faces with YuNet (OpenCV DNN face detector).
  2. Align and crop the largest face.
  3. Turn it into a 128-number "face fingerprint" (embedding) with SFace.
  4. Compare two fingerprints with cosine similarity.

Both models come from the official OpenCV model zoo and run on CPU.
"""

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "detector": ("face_detection_yunet_2023mar.onnx",
                 _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "recognizer": ("face_recognition_sface_2021dec.onnx",
                   _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}

# Cosine-similarity thresholds.
# 0.363 is the threshold published with SFace for "same person".
# Scores between the two thresholds are deliberately left for a human to decide.
SAME_PERSON_THRESHOLD = 0.363
STRONG_MATCH_THRESHOLD = 0.50


@dataclass
class FaceResult:
    box: tuple          # (x, y, w, h) of the chosen face
    faces_found: int    # how many faces were detected in the image
    embedding: np.ndarray


@dataclass
class Verdict:
    score: float        # cosine similarity, roughly -1..1 (higher = more similar)
    label: str          # "match", "review", or "no_match"
    message: str


def ensure_models():
    """Download the two ONNX models on first run."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    for key, (name, url) in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path) or os.path.getsize(path) < 10_000:
            print(f"[*] Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


class FaceEngine:
    def __init__(self, det_score_threshold=0.8):
        paths = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            paths["detector"], "", (320, 320), det_score_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(paths["recognizer"], "")

    def analyze(self, image):
        """Return a FaceResult for the largest face in a BGR image, or None."""
        if image is None:
            raise ValueError("Image could not be read.")
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        # Pick the largest face (by box area)
        face = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(image, face)
        embedding = self.recognizer.feature(aligned)
        box = tuple(int(v) for v in face[:4])
        return FaceResult(box=box, faces_found=len(faces), embedding=embedding)

    def compare(self, face_a, face_b):
        """Compare two FaceResults and return a Verdict."""
        score = float(self.recognizer.match(
            face_a.embedding, face_b.embedding, cv2.FaceRecognizerSF_FR_COSINE))

        if score >= STRONG_MATCH_THRESHOLD:
            return Verdict(score, "match", "Likely the same person")
        if score >= SAME_PERSON_THRESHOLD:
            return Verdict(score, "review", "Possible match — needs human review")
        return Verdict(score, "no_match", "Likely different people")


def load_image(path):
    """cv2.imread that also works with non-ASCII (e.g. Arabic) file paths."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a readable image: {os.path.basename(path)}")
    return image
