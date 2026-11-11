"""
Visual Match Detector — face verification across time.

Compare a face in an old photo with a face in a recent photo and estimate
whether they belong to the same person.

Usage:
    python app.py                    # desktop app
    python app.py old.jpg new.jpg    # command line
"""

import sys
import threading

import cv2

from face_engine import FaceEngine, load_image

PREVIEW_W, PREVIEW_H = 300, 300
COLORS = {"match": "#3fb950", "review": "#d29922", "no_match": "#f85149"}


def fit_to_box(image, max_w, max_h):
    """Resize keeping the aspect ratio, so faces are never stretched."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_face_box(image, face, color_bgr):
    out = image.copy()
    if face is not None:
        x, y, w, h = face.box
        thickness = max(2, image.shape[1] // 200)
        cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, thickness)
    return out


# ---------------------------------------------------------------- command line
def format_scores(verdict):
    parts = []
    if verdict.arcface_score is not None:
        parts.append(f"ArcFace {verdict.arcface_score:.3f}")
    parts.append(f"SFace {verdict.sface_score:.3f}")
    return "  ·  ".join(parts)


def run_cli(path_a, path_b):
    engine = FaceEngine()
    if engine.arcface is None:
        print(f"[!] ArcFace unavailable, using SFace only ({engine.arcface_error})")
    faces = []
    for path in (path_a, path_b):
        try:
            image = load_image(path)
        except (ValueError, OSError) as exc:
            print(f"[-] {exc}")
            return 1
        face = engine.analyze(image)
        if face is None:
            print(f"[-] No face found in {path}")
            return 2
        if face.faces_found > 1:
            print(f"[!] {path}: {face.faces_found} faces found, using the largest one")
        faces.append(face)

    verdict = engine.compare(*faces)
    print(f"[+] Scores  : {format_scores(verdict)}")
    print(f"[+] Verdict : {verdict.message}")
    return 0


# ---------------------------------------------------------------- desktop app
def run_gui():
    from tkinter import Tk, Label, Button, Canvas, filedialog
    from PIL import Image, ImageTk

    BG, PANEL, FG = "#0d1117", "#161b22", "#e6edf3"

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Visual Match Detector")
            root.configure(bg=BG)
            root.resizable(False, False)

            Label(root, text="Compare a face across two photos",
                  font=("Helvetica", 14, "bold"), fg=FG, bg=BG).pack(pady=(12, 6))

            btns = Label(root, bg=BG)
            btns.pack()
            Button(btns, text="Load photo 1 (older)", width=20,
                   command=lambda: self.load(0)).pack(side="left", padx=6)
            Button(btns, text="Load photo 2 (recent)", width=20,
                   command=lambda: self.load(1)).pack(side="left", padx=6)

            self.canvas = Canvas(root, width=PREVIEW_W * 2 + 30, height=PREVIEW_H + 20,
                                 bg=PANEL, highlightthickness=0)
            self.canvas.pack(padx=12, pady=10)

            self.compare_btn = Button(root, text="Compare faces", width=20,
                                      state="disabled", command=self.start_compare)
            self.compare_btn.pack()

            self.result = Label(root, text="Loading face models... (first run downloads ~330 MB)", font=("Helvetica", 12),
                                fg=FG, bg=BG)
            self.result.pack(pady=(8, 14))

            self.images = [None, None]
            self.faces = [None, None]
            self._tk_refs = [None, None]
            self.engine = None
            threading.Thread(target=self._load_engine, daemon=True).start()

        # Model loading happens in the background; UI updates go through root.after
        def _load_engine(self):
            try:
                engine = FaceEngine()
                self.root.after(0, self._engine_ready, engine, None)
            except Exception as exc:
                self.root.after(0, self._engine_ready, None, exc)

        def _engine_ready(self, engine, error):
            if error:
                self.result.config(text=f"Could not load models: {error}", fg=COLORS["no_match"])
                return
            self.engine = engine
            text = "Load two photos to start."
            if engine.arcface is None:
                text += "\n(ArcFace unavailable — using SFace only)"
            self.result.config(text=text, fg=FG)
            self._refresh_button()

        def load(self, slot):
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
            if not path:
                return
            try:
                self.images[slot] = load_image(path)
            except ValueError as exc:
                self.result.config(text=str(exc), fg=COLORS["no_match"])
                return
            self.faces[slot] = None
            self.show(slot, color_bgr=(255, 255, 255))
            self.result.config(text="", fg=FG)
            self._refresh_button()

        def show(self, slot, color_bgr):
            image = draw_face_box(self.images[slot], self.faces[slot], color_bgr)
            image = fit_to_box(image, PREVIEW_W, PREVIEW_H)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._tk_refs[slot] = tk_img          # keep a reference or Tk drops it

            tag = f"slot{slot}"
            self.canvas.delete(tag)               # clear only this slot
            x = 10 + slot * (PREVIEW_W + 10) + PREVIEW_W // 2
            self.canvas.create_image(x, 10 + PREVIEW_H // 2, image=tk_img, tags=tag)

        def _refresh_button(self):
            ready = self.engine is not None and all(img is not None for img in self.images)
            self.compare_btn.config(state="normal" if ready else "disabled")

        def start_compare(self):
            self.compare_btn.config(state="disabled")
            self.result.config(text="Analyzing faces...", fg=FG)
            threading.Thread(target=self._compare_worker, daemon=True).start()

        def _compare_worker(self):
            try:
                faces = [self.engine.analyze(img) for img in self.images]
                verdict = None
                if all(faces):
                    verdict = self.engine.compare(*faces)
                self.root.after(0, self._show_result, faces, verdict, None)
            except Exception as exc:
                self.root.after(0, self._show_result, None, None, exc)

        def _show_result(self, faces, verdict, error):
            self._refresh_button()
            if error:
                self.result.config(text=f"Error: {error}", fg=COLORS["no_match"])
                return

            missing = [str(i + 1) for i, f in enumerate(faces) if f is None]
            if missing:
                self.result.config(text=f"No face found in photo {', '.join(missing)}.",
                                   fg=COLORS["no_match"])
                return

            self.faces = faces
            hex_color = COLORS[verdict.label]
            bgr = tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))
            for slot in (0, 1):
                self.show(slot, bgr)

            note = ""
            extra = [f"photo {i + 1}" for i, f in enumerate(faces) if f.faces_found > 1]
            if extra:
                note = f"\n(More than one face in {', '.join(extra)} — used the largest)"
            self.result.config(
                text=f"{verdict.message}\n{format_scores(verdict)}{note}", fg=hex_color)

    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sys.exit(run_cli(sys.argv[1], sys.argv[2]))
    if len(sys.argv) == 1:
        run_gui()
    else:
        print(__doc__)
        sys.exit(1)
