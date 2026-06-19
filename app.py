from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import torch
import numpy as np
import time
import json

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

# -------------------------------
# Load Zero-DCE Model
# -------------------------------
from model.model import enhance_net_nopool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Zero-DCE model...")
dce_model = enhance_net_nopool().to(device)
dce_model.load_state_dict(torch.load("model/weights.pth", map_location=device))
dce_model.eval()
print("Model loaded successfully.")

# Global progress tracker
processing_status = {"progress": 0, "status": "idle", "stats": {}}

# -------------------------------
# Enhancement Functions
# -------------------------------
def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_frame(frame):
    """Fast denoising using bilateral filter — preserves edges better than Gaussian."""
    return cv2.bilateralFilter(frame, d=7, sigmaColor=75, sigmaSpace=75)


def zero_dce_enhance(frame):
    """Enhance frame using Zero-DCE deep learning model."""
    h, w = frame.shape[:2]
    # Zero-DCE works best with dimensions divisible by 4
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    if pad_h or pad_w:
        frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = dce_model(tensor)

    enhanced = (output[0] if isinstance(output, tuple) else output)
    enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # Remove padding
    return enhanced[:h, :w]


def compute_brightness(frame):
    """Return mean brightness of a frame (0–255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_psnr(original, enhanced):
    """Compute PSNR between two frames."""
    mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

# -------------------------------
# Improved Motion Detection
# (MOG2 + optical-flow hybrid)
# -------------------------------
class MotionDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=40,
            detectShadows=False
        )
        self.prev_gray = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.motion_count = 0

    def detect(self, frame):
        """Returns (annotated_frame, motion_detected: bool, num_detections: int)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()
        detections = 0

        # --- MOG2 mask ---
        mog_mask = self.fgbg.apply(frame)
        mog_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.kernel)
        mog_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_DILATE, self.kernel, iterations=2)

        # --- Optical flow confirmation (reduces false positives) ---
        flow_mask = np.zeros_like(mog_mask)
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=13,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Threshold: only keep pixels with significant motion
            flow_mask = (mag > 1.5).astype(np.uint8) * 255

        # Combine masks (AND logic reduces noise)
        if self.prev_gray is not None:
            combined = cv2.bitwise_and(mog_mask, flow_mask)
        else:
            combined = mog_mask

        self.prev_gray = gray.copy()

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 25 or h < 25:
                continue

            # Aspect ratio filter (remove very thin/tall false positives)
            aspect = w / h
            if aspect > 6 or aspect < 0.15:
                continue

            detections += 1
            self.motion_count += 1

            # Classify by size
            if area > 15000:
                label, color = "Vehicle", (0, 100, 255)
            elif area > 5000:
                label, color = "Person", (0, 255, 0)
            else:
                label, color = "Object", (0, 220, 220)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Background-filled label for readability
            text = f"{label} ({int(area)}px)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 6, y), color, -1)
            cv2.putText(annotated, text, (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated, detections > 0, detections


# -------------------------------
# HUD overlay
# -------------------------------
def draw_hud(frame, frame_num, fps, brightness_orig, brightness_enh, motion, detections):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    status_color = (0, 80, 255) if motion else (0, 200, 60)
    status_text = f"MOTION ALERT x{detections}" if motion else "CLEAR"
    psnr = compute_psnr(frame[:h, :w//2], frame[:h, w//2:]) if w > 640 else 0

    info = (f"Frame:{frame_num}  FPS:{fps:.1f}  "
            f"Orig Lum:{brightness_orig:.0f}  "
            f"Enh Lum:{brightness_enh:.0f}  "
            f"Gain:{brightness_enh - brightness_orig:+.0f}")

    cv2.putText(frame, info, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, status_text, (w - 230, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # Divider line between original and enhanced halves
    if w > 640:
        mid = w // 2
        cv2.line(frame, (mid, 0), (mid, h), (255, 255, 255), 1)
        cv2.putText(frame, "ORIGINAL", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "ENHANCED", (mid + 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1)
    return frame


# -------------------------------
# Video Processing
# -------------------------------
def process_video(input_path, output_path):
    global processing_status

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target resolution: keep aspect ratio, max width 640
    target_w = min(640, src_w)
    target_h = int(src_h * target_w / src_w) if src_w > 0 else 480
    output_w = target_w * 2  # side-by-side

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (output_w, target_h))

    detector = MotionDetector()

    stats = {
        "total_frames": total_frames,
        "motion_frames": 0,
        "avg_brightness_orig": 0,
        "avg_brightness_enh": 0,
        "processing_time": 0,
    }

    frame_num = 0
    total_brightness_orig = 0.0
    total_brightness_enh = 0.0
    t_start = time.time()

    processing_status["status"] = "processing"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        frame = cv2.resize(frame, (target_w, target_h))
        original = frame.copy()

        # ---- Enhancement pipeline ----
        enhanced = zero_dce_enhance(frame)
        enhanced = denoise_frame(enhanced)
        enhanced = gamma_correction(enhanced, gamma=1.4)
        enhanced = apply_clahe(enhanced)

        kernel_sharp = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharp)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        # ---- Motion detection ----
        annotated, motion_detected, n_detections = detector.detect(enhanced)

        if motion_detected:
            stats["motion_frames"] += 1

        b_orig = compute_brightness(original)
        b_enh = compute_brightness(enhanced)
        total_brightness_orig += b_orig
        total_brightness_enh += b_enh

        elapsed = time.time() - t_start
        fps_live = frame_num / elapsed if elapsed > 0 else 0

        # ---- HUD ----
        combined = cv2.hconcat([original, annotated])
        combined = draw_hud(combined, frame_num, fps_live,
                            b_orig, b_enh, motion_detected, n_detections)

        out.write(combined)

        processing_status["progress"] = int((frame_num / max(total_frames, 1)) * 100)
        processing_status["current_frame"] = frame_num

    cap.release()
    out.release()

    stats["processing_time"] = round(time.time() - t_start, 2)
    stats["avg_brightness_orig"] = round(total_brightness_orig / max(frame_num, 1), 1)
    stats["avg_brightness_enh"] = round(total_brightness_enh / max(frame_num, 1), 1)
    stats["brightness_gain"] = round(stats["avg_brightness_enh"] - stats["avg_brightness_orig"], 1)
    stats["motion_percent"] = round(stats["motion_frames"] / max(frame_num, 1) * 100, 1)
    stats["total_frames_processed"] = frame_num

    processing_status["stats"] = stats
    processing_status["status"] = "converting"

    # Convert to MP4 for browser playback
    mp4_path = output_path.replace(".avi", ".mp4")
    os.system(f'ffmpeg -y -i "{output_path}" -vcodec libx264 -crf 23 "{mp4_path}"')

    processing_status["status"] = "done"
    return mp4_path, stats


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global processing_status
    processing_status = {"progress": 0, "status": "starting", "stats": {}}

    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    output_avi = os.path.join(app.config["OUTPUT_FOLDER"], "output.avi")
    mp4_out = output_avi.replace(".avi", ".mp4")

    file.save(input_path)

    for p in [output_avi, mp4_out]:
        if os.path.exists(p):
            os.remove(p)

    mp4_path, stats = process_video(input_path, output_avi)

    return jsonify({
        "video_url": "/" + mp4_path,
        "stats": stats
    })


@app.route("/progress")
def progress():
    """SSE endpoint for live progress updates."""
    def stream():
        while processing_status.get("status") not in ("done", "idle", "error"):
            data = json.dumps(processing_status)
            yield f"data: {data}\n\n"
            time.sleep(0.5)
        yield f"data: {json.dumps(processing_status)}\n\n"

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
