"""
laptop_vision.py  -  laptop webcam -> ESP32 (USB serial)   (MediaPipe Tasks API)
                     + dataset logger for Edge Impulse (TinyML training)

Needs in the SAME folder:
  face_landmarker.task
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  efficientdet_lite0.tflite   (optional, for phone detection)
    https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite

Sends to the ESP32 about 14 times per second:
      D,<ear>,<face 0/1>,<yaw>,<pitch>,<lowVis 0/1>,<phone 0/1/2>\n
        phone: 0 = none, 1 = phone at ear, 2 = phone in front / low
Sends  C\n  when you press 'c'.
Reads from the ESP32:
      S,...  status    M,...  debug    P,1  calibration started (button or key)
      P,2    nod phase  K,x  calibration result (1 done, 0 failed, 2 camera-only)
      F,...  feature line (5 Hz) -> saved to CSV while recording

Keys:  c = calibrate     g = glasses mode     Esc = quit
       Recording (press the same key again, or 0, to stop):
         1 attentive   2 partial   3 drowsy   4 distracted   5 phonetalk   6 phoneuse
       Files are saved in the "dataset" folder as  <label>.take<N>.csv
       (Edge Impulse takes the label from the filename before the first dot)

Install:  pip install opencv-python mediapipe pyserial numpy
Close the Arduino Serial Monitor before running.
"""
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import serial
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ------------------------------ settings ------------------------------
PORT = "COM5"
BAUD = 115200
CAM_INDEX = 0
SEND_PERIOD = 0.07
CAL_SECONDS = 3.0
LOW_LIGHT_GRAY = 70
YAW_SIGN = 1.0           # ESP32 needs: looking RIGHT = positive yaw
PITCH_SIGN = 1.0         # ESP32 needs: looking DOWN  = positive pitch
PHONE_EVERY_N = 3        # run phone detector every N frames (saves CPU)
PHONE_SCORE = 0.35
PHONE_HOLD_S = 0.4       # keep last phone result this long between detections

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "face_landmarker.task")
PHONE_MODEL_PATH = os.path.join(BASE, "efficientdet_lite0.tflite")

# ------------------------------ data logger ------------------------------
LABEL_KEYS = {ord('1'): "attentive", ord('2'): "partial", ord('3'): "drowsy",
              ord('4'): "distracted", ord('5'): "phonetalk", ord('6'): "phoneuse"}
LOG_DIR = os.path.join(BASE, "dataset")
os.makedirs(LOG_DIR, exist_ok=True)
# first 9 values of the ESP32 "F" line (the timers, mode and state are NOT logged)
FEAT_NAMES = ["earRatio", "yaw", "pitch", "fwd", "side", "face", "mpuCal", "lowVis", "phone"]
ST_CALIBRATING = "7"     # state code in the F line while the ESP32 is calibrating

R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]

POSE_IDX = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, 63.6, 12.5),
    (-43.3, -32.7, 26.0),
    (43.3, -32.7, 26.0),
    (-28.9, 28.9, 24.1),
    (28.9, 28.9, 24.1)], dtype=np.float64)

# face landmarks used for the phone zone
LM_EAR_L, LM_EAR_R = 234, 454      # sides of the face (ear area)
LM_FOREHEAD, LM_CHIN, LM_NOSE = 10, 152, 1


def eye_aspect_ratio(pts, idx):
    p = [np.array(pts[i]) for i in idx]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / \
           (2.0 * np.linalg.norm(p[0] - p[3]))


def head_pose(pts, w, h):
    img = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw = float(angles[0]), float(angles[1])
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return YAW_SIGN * yaw, PITCH_SIGN * pitch


def phone_zone_from_box(box, pts):
    """box = (x, y, w, h) in pixels. pts = face landmarks in pixels or None.
    Returns 1 (at ear), 2 (in front / low)."""
    cx = box[0] + box[2] / 2
    cy = box[1] + box[3] / 2
    if pts is None:
        return 2                                    # face lost + phone visible -> using
    face_w = np.linalg.norm(np.array(pts[LM_EAR_L]) - np.array(pts[LM_EAR_R]))
    top_y = pts[LM_FOREHEAD][1]
    chin_y = pts[LM_CHIN][1]
    for ear in (pts[LM_EAR_L], pts[LM_EAR_R]):
        d = np.hypot(cx - ear[0], cy - ear[1])
        if d < 0.8 * face_w and top_y - 0.2 * face_w < cy < chin_y + 0.1 * face_w:
            return 1                                # beside the head -> talking
    if cy > pts[LM_NOSE][1]:                        # below nose level -> using
        return 2
    return 0


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found:", MODEL_PATH)
        sys.exit(1)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0)
    except serial.SerialException as e:
        print("Cannot open", PORT, "->", e)
        sys.exit(1)
    time.sleep(2)
    ser.reset_input_buffer()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    phone_det = None
    if os.path.exists(PHONE_MODEL_PATH):
        popt = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=PHONE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            score_threshold=PHONE_SCORE,
            category_allowlist=["cell phone"],
            max_results=3)
        phone_det = vision.ObjectDetector.create_from_options(popt)
    else:
        print("WARNING: efficientdet_lite0.tflite not found -> phone detection OFF (sending 0)")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Webcam not found. Try CAM_INDEX = 1.")
        ser.close()
        sys.exit(1)

    ear_hist = deque(maxlen=3)
    yaw_s, pitch_s = 0.0, 0.0
    yaw0, pitch0 = 0.0, 0.0
    open_ear = 0.27                                 # only used for display now
    bright = 128.0
    glasses_mode = False

    cal_until = 0.0
    cal_ears, cal_yaws, cal_pitches = [], [], []

    rx = b""
    esp_status = "waiting for ESP32..."
    esp_debug = ""
    esp_cal = ""
    last_send = 0.0
    t0 = time.monotonic()
    last_ts = -1
    last_pts_ts = -1
    frame_no = 0
    phone_zone = 0
    phone_seen_t = 0.0

    # ---- recording state ----
    rec_file = None
    rec_writer = None
    rec_label = ""
    rec_t0 = 0.0
    rec_rows = 0
    take_no = int(time.time()) % 100000

    def start_camera_calibration():
        nonlocal cal_until, cal_ears, cal_yaws, cal_pitches
        cal_until = time.monotonic() + CAL_SECONDS
        cal_ears, cal_yaws, cal_pitches = [], [], []

    def stop_recording():
        nonlocal rec_file, rec_writer, rec_label, rec_rows
        if rec_file:
            rec_file.close()
            print(f"Saved take: {rec_label}, {rec_rows} rows ({rec_rows / 5:.0f} s)")
        rec_file, rec_writer, rec_label, rec_rows = None, None, "", 0

    def start_recording(label):
        nonlocal rec_file, rec_writer, rec_label, rec_t0, rec_rows, take_no
        stop_recording()
        take_no += 1
        # Edge Impulse reads the label from the filename before the first dot
        path = os.path.join(LOG_DIR, f"{label}.take{take_no}.csv")
        rec_file = open(path, "w", newline="")
        rec_writer = csv.writer(rec_file)
        rec_writer.writerow(["timestamp"] + FEAT_NAMES)
        rec_label = label
        rec_rows = 0
        rec_t0 = time.monotonic()
        print("Recording:", label, "->", path)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam read failed")
                break
            h, w = frame.shape[:2]
            frame_no += 1

            gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[::8, ::8]))
            bright = 0.95 * bright + 0.05 * gray_mean
            low_vis = 1 if (glasses_mode or bright < LOW_LIGHT_GRAY) else 0

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - t0) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            face = 0
            ear_s = 0.0
            pts = None
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                pts = [(p.x * w, p.y * h) for p in lm]
                ear_now = (eye_aspect_ratio(pts, R_EYE) + eye_aspect_ratio(pts, L_EYE)) / 2
                ear_hist.append(ear_now)
                ear_s = sum(ear_hist) / len(ear_hist)
                pose = head_pose(pts, w, h)
                if pose is not None:
                    yaw_s = 0.7 * yaw_s + 0.3 * pose[0]
                    pitch_s = 0.7 * pitch_s + 0.3 * pose[1]
                face = 1
            else:
                ear_hist.clear()

            # ---- phone detection (every N frames, result held briefly) ----
            if phone_det is not None and frame_no % PHONE_EVERY_N == 0:
                pts_ts = max(ts, last_pts_ts + 1)
                last_pts_ts = pts_ts
                pres = phone_det.detect_for_video(mp_image, pts_ts)
                if pres.detections:
                    best = max(pres.detections, key=lambda d: d.categories[0].score)
                    bb = best.bounding_box
                    box = (bb.origin_x, bb.origin_y, bb.width, bb.height)
                    phone_zone = phone_zone_from_box(box, pts)
                    phone_seen_t = time.monotonic()
                    cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                  (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255), 2)
                elif time.monotonic() - phone_seen_t > PHONE_HOLD_S:
                    phone_zone = 0
            if phone_det is None:
                phone_zone = 0

            # ---- camera calibration (straight pose = neutral) ----
            calibrating = time.monotonic() < cal_until
            if calibrating and face:
                cal_ears.append(ear_s)
                cal_yaws.append(yaw_s)
                cal_pitches.append(pitch_s)
            elif cal_until and not calibrating:
                cal_until = 0.0
                if len(cal_ears) >= 10:
                    open_ear = float(np.clip(np.percentile(cal_ears, 70), 0.18, 0.40))
                    yaw0 = float(np.median(cal_yaws))
                    pitch0 = float(np.median(cal_pitches))
                    print(f"Camera calibrated: open EAR {open_ear:.3f}, "
                          f"yaw0 {yaw0:.1f}, pitch0 {pitch0:.1f}")
                else:
                    print("Camera calibration failed: face not seen, try again")

            yaw_out = yaw_s - yaw0
            pitch_out = pitch_s - pitch0

            # ---- send to ESP32 (matches the .ino parser exactly) ----
            now = time.monotonic()
            if now - last_send >= SEND_PERIOD:
                if face:
                    line = f"D,{ear_s:.3f},1,{yaw_out:.1f},{pitch_out:.1f},{low_vis},{phone_zone}\n"
                else:
                    line = f"D,0.000,0,0.0,0.0,{low_vis},{phone_zone}\n"
                ser.write(line.encode())
                last_send = now

            # ---- read from ESP32 ----
            data = ser.read(1024)
            if data:
                rx += data
                while b"\n" in rx:
                    raw, rx = rx.split(b"\n", 1)
                    msg = raw.decode(errors="ignore").strip()
                    if msg.startswith("S,"):
                        esp_status = msg[2:]
                    elif msg.startswith("M,"):
                        esp_debug = msg[2:]
                        print(msg)
                    elif msg.startswith("F,") and rec_writer:
                        parts = msg[2:].split(",")
                        # 17 values; the last one is the ESP32 state -> skip while calibrating
                        if len(parts) == 17 and parts[16] != ST_CALIBRATING:
                            rec_writer.writerow([int((time.monotonic() - rec_t0) * 1000)]
                                                + parts[:9])
                            rec_rows += 1
                    elif msg == "P,1":              # ESP32 calibration started (button or key)
                        esp_cal = "CAL 1: sit straight"
                        if not (time.monotonic() < cal_until):
                            start_camera_calibration()
                    elif msg == "P,2":
                        esp_cal = "CAL 2: nod forward"
                    elif msg.startswith("K,"):
                        esp_cal = {"K,1": "calibration DONE", "K,0": "calibration FAILED",
                                   "K,2": "camera-only calibration DONE"}.get(msg, "")
                        print(esp_cal)

            # ---- draw ----
            cv2.putText(frame, f"EAR {ear_s:.2f}  face {face}  phone {phone_zone}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw {yaw_out:.0f}  Pitch {pitch_out:.0f}  "
                               f"light {bright:.0f}  lowVis {low_vis}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if calibrating:
                cv2.putText(frame, "CALIBRATING: look straight, eyes open",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            if esp_cal:
                cv2.putText(frame, esp_cal, (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if rec_label:
                cv2.circle(frame, (w - 30, 28), 9, (0, 0, 255), -1)
                cv2.putText(frame, f"REC: {rec_label}  {rec_rows / 5:.0f}s",
                            (w - 330, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "ESP32: " + esp_status, (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(frame, esp_debug[:90], (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("webcam -> ESP32  (c cal, g glasses, 1-6 record, 0 stop, Esc quit)", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord("c"):
                ser.write(b"C\n")
                start_camera_calibration()
            elif k == ord("g"):
                glasses_mode = not glasses_mode
                print("Glasses mode:", glasses_mode)
            elif k in LABEL_KEYS:
                if rec_label == LABEL_KEYS[k]:
                    stop_recording()                # same key again = stop
                else:
                    start_recording(LABEL_KEYS[k])
            elif k == ord("0"):
                stop_recording()
    finally:
        stop_recording()
        cap.release()
        ser.close()
        landmarker.close()
        if phone_det is not None:
            phone_det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()