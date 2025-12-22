import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk
import threading

class ImageComparator:
    def __init__(self, master):
        self.master = master
        master.title("Visual Match Detector")
        master.configure(bg="#2e2e2e")

        self.label = Label(master, text="Upload two images to compare:",
                           font=("Helvetica", 14, "bold"),
                           fg="white", bg="#2e2e2e")
        self.label.pack()

        self.upload_btn1 = Button(master, text="Upload Image 1",
                                  font=("Arial", 12),
                                  bg="#4d4d4d", fg="white",
                                  command=self.load_image1)
        self.upload_btn1.pack()

        self.upload_btn2 = Button(master, text="Upload Image 2",
                                  font=("Arial", 12),
                                  bg="#4d4d4d", fg="white",
                                  command=self.load_image2)
        self.upload_btn2.pack()

        self.canvas = Canvas(master, width=620, height=320, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack()

        self.compare_btn = Button(master, text="🔍 Compare Images",
                                  font=("Arial", 12, "bold"),
                                  bg="#6c6c6c", fg="white",
                                  command=self.start_comparison,
                                  state="disabled")
        self.compare_btn.pack(pady=10)

        self.result_label = Label(master, text="", font=("Arial", 12), bg="#2e2e2e", fg="white")
        self.result_label.pack()

        self.image1 = None
        self.image2 = None

    def load_image1(self):
        path = filedialog.askopenfilename()
        if path:
            self.image1 = cv2.imread(path)
            

            # Display after loading
            img_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img_pil)

            self.canvas.create_image(10, 10, anchor="nw", image=img_tk)
            self.canvas.image1 = img_tk

            self.check_ready()

    def load_image2(self):
        path = filedialog.askopenfilename()
        if path:
            self.image2 = cv2.imread(path)
            
       

            # Display after loading
            img_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img_pil)

            self.canvas.create_image(310, 10, anchor="nw", image=img_tk)
            self.canvas.image2 = img_tk

            self.check_ready()

    def check_ready(self):
        if self.image1 is not None and self.image2 is not None:
            self.compare_btn.config(state="normal")

    def start_comparison(self):
        self.result_label.config(text="🔄 Processing...")
        threading.Thread(target=self.compare_images).start()

    def compare_images(self):
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            self.result_label.config(text="❌ No features detected.")
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if matrix is not None:
                h, w = self.image1.shape[:2]
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                image2_copy = self.image2.copy()
                cv2.polylines(image2_copy, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                result = cv2.drawMatches(self.image1, kp1, image2_copy, kp2, good_matches[:20], None, flags=2)
                status = "🟢 Region Match Detected"
            else:
                result = cv2.drawMatches(self.image1, kp1, self.image2, kp2, good_matches[:20], None, flags=2)
                status = "🟡 Matches Found, No Region Detected"
        else:
            result = cv2.drawMatches(self.image1, kp1, self.image2, kp2, good_matches, None, flags=2)
            status = "🔴 No Significant Match"

        result = cv2.resize(result, (620, 300))
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        result_tk = ImageTk.PhotoImage(result_pil)

        self.canvas.create_image(0, 0, anchor="nw", image=result_tk)
        self.canvas.image = result_tk
        self.result_label.config(text=f"✅ Matches: {len(good_matches)} — {status}")


# Run the app
root = Tk()
app = ImageComparator(root)
root.mainloop()
