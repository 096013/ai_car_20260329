# -*- coding: utf-8 -*-
import atexit
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

try:
    from picamera2 import Picamera2, libcamera
except ImportError:  # pragma: no cover - fallback for non-RPi environments
    Picamera2 = None
    libcamera = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - fallback for environments without YOLO
    YOLO = None

import motor_control

app = Flask(__name__)

WIDTH, HEIGHT = 640, 480
YOLO_MODEL_PATH = "yolo_model.pt"
TRAFFIC_LIGHT_LABELS = {"traffic light", "red_light", "red-light", "red light"}
LANE_COLOR_RANGES = {
    "yellow": [((15, 80, 80), (40, 255, 255))],
    "blue": [((90, 80, 60), (135, 255, 255))],
    "white": [((0, 0, 170), (180, 70, 255))],
}


class PIDController:
    def __init__(self, kp=0.35, ki=0.0, kd=0.18):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        safe_dt = max(dt, 1e-3)
        self.integral += error * safe_dt
        derivative = (error - self.prev_error) / safe_dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class DummyCamera:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height

    def capture_array(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Camera not available",
            (70, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def stop(self):
        return None


class OpenCVCamera:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.cap = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture_array(self):
        ok, frame = self.cap.read()
        if not ok:
            return DummyCamera(self.width, self.height).capture_array()
        return frame

    def stop(self):
        if self.cap:
            self.cap.release()


def create_camera():
    if Picamera2 is not None and libcamera is not None:
        camera = Picamera2()
        transform = libcamera.Transform(hflip=True, vflip=True)
        config = camera.create_preview_configuration(
            main={"size": (WIDTH, HEIGHT)},
            transform=transform,
        )
        camera.configure(config)
        camera.start()
        return camera

    fallback = OpenCVCamera(WIDTH, HEIGHT)
    if fallback.cap.isOpened():
        return fallback
    return DummyCamera(WIDTH, HEIGHT)


def load_yolo_model():
    if YOLO is None or not os.path.exists(YOLO_MODEL_PATH):
        return None
    try:
        return YOLO(YOLO_MODEL_PATH)
    except Exception:
        return None


class NullRobotControl:
    available = False

    def move(self, action, speed=None):
        return None

    def drive(self, forward_speed, turn_rate=0):
        return None

    def stop(self):
        return None


motor_error_message = ""


def create_car():
    global motor_error_message
    try:
        controller = motor_control.RobotControl()
        controller.available = True
        motor_error_message = ""
        return controller
    except Exception as exc:
        motor_error_message = str(exc)
        return NullRobotControl()


car = create_car()
camera = create_camera()
model = load_yolo_model()
pid = PIDController()

state_lock = threading.Lock()
frame_lock = threading.Lock()
stream_jpeg = None
running = True
lane_memory = {
    "last_error": 0.0,
    "last_seen": 0.0,
    "lane_width": WIDTH * 0.42,
}

system_state = {
    "mode": "manual",
    "speed": 0.70,
    "lane_color": "yellow",
    "pid": {"kp": 0.35, "ki": 0.0, "kd": 0.18},
    "ai_message": "系統啟動中",
    "objects": [],
    "red_light": False,
    "lane_detected": False,
    "lane_offset": 0.0,
    "last_command": "stop",
    "yolo_enabled": model is not None,
    "motor_enabled": getattr(car, "available", False),
    "motor_error": motor_error_message,
    "control_debug": "idle",
}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_speed_value(raw_speed):
    speed = float(raw_speed)
    if speed > 1:
        speed = speed / 100.0
    return clamp(speed, 0.0, 1.0)


def speed_to_pwm(speed_ratio):
    ratio = clamp(float(speed_ratio), 0.0, 1.0)
    if ratio <= 0:
        return 0

    pwm = int(round(ratio * 100))
    # Many classroom chassis cannot overcome static friction below ~35% PWM.
    return max(35, pwm)


def get_state_snapshot():
    with state_lock:
        return {
            "mode": system_state["mode"],
            "speed": system_state["speed"],
            "lane_color": system_state["lane_color"],
            "pid": dict(system_state["pid"]),
            "ai_message": system_state["ai_message"],
            "objects": list(system_state["objects"]),
            "red_light": system_state["red_light"],
            "lane_detected": system_state["lane_detected"],
            "lane_offset": system_state["lane_offset"],
            "last_command": system_state["last_command"],
            "yolo_enabled": system_state["yolo_enabled"],
            "motor_enabled": system_state["motor_enabled"],
            "motor_error": system_state["motor_error"],
            "control_debug": system_state["control_debug"],
        }


def update_state(**kwargs):
    with state_lock:
        for key, value in kwargs.items():
            if key == "pid":
                system_state["pid"] = dict(value)
            else:
                system_state[key] = value


def build_lane_mask(hsv_frame, lane_color):
    if lane_color == "white":
        return build_white_lane_mask(cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR))

    ranges = LANE_COLOR_RANGES.get(lane_color, LANE_COLOR_RANGES["yellow"])
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv_frame, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def build_white_lane_mask(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    # White lane: high lightness, low saturation.
    white_mask = cv2.inRange(lightness, 145, 255) & cv2.inRange(saturation, 0, 145)

    # Suppress overexposed glare regions before line extraction.
    glare_mask = cv2.inRange(lightness, 245, 255) & cv2.inRange(saturation, 0, 90)
    glare_mask = cv2.GaussianBlur(glare_mask, (9, 9), 0)
    _, glare_mask = cv2.threshold(glare_mask, 100, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(glare_mask))

    # Use local contrast to separate white tape from bright floor reflections.
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast = cv2.subtract(gray, blur)
    contrast_mask = cv2.inRange(contrast, 6, 255)

    # Edge detection is more stable for white lanes on glossy floors.
    edges = cv2.Canny(gray, 50, 140)
    edge_mask = cv2.bitwise_and(edges, white_mask)
    edge_mask = cv2.bitwise_and(edge_mask, contrast_mask)

    line_mask = np.zeros_like(edge_mask)
    lines = cv2.HoughLinesP(
        edge_mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=24,
        minLineLength=45,
        maxLineGap=18,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 45:
                continue

            # Ignore near-horizontal edges from floor texture.
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle < 25:
                continue

            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 10)

    mask = cv2.bitwise_or(line_mask, cv2.bitwise_and(white_mask, contrast_mask))

    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_large = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    mask = cv2.dilate(mask, kernel_small, iterations=2)
    return mask


def build_white_birdeye(mask):
    height, width = mask.shape[:2]
    src = np.float32([
        [width * 0.18, height * 0.06],
        [width * 0.82, height * 0.06],
        [width * 0.98, height * 0.98],
        [width * 0.02, height * 0.98],
    ])
    dst = np.float32([
        [width * 0.18, 0],
        [width * 0.82, 0],
        [width * 0.82, height - 1],
        [width * 0.18, height - 1],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(mask, matrix, (width, height))
    kernel = np.ones((5, 5), dtype=np.uint8)
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)
    warped = cv2.morphologyEx(warped, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return warped, inverse


def map_birdeye_point(point, inverse_matrix, roi_top):
    points = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, inverse_matrix)[0][0]
    return int(mapped[0]), int(mapped[1] + roi_top)


def extract_white_lane_single_contour(frame, mask, roi_top):
    roi = mask[roi_top:, :]
    overlay = frame.copy()
    height, width = frame.shape[:2]
    center_x = width // 2
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        major = max(rect_w, rect_h)
        minor = min(rect_w, rect_h)
        if minor < 1:
            continue

        aspect_ratio = major / minor
        if aspect_ratio < 2.4 or major < 80:
            continue

        score = area * min(aspect_ratio, 8.0)
        if score > best_score:
            best = contour
            best_score = score

    if best is None:
        return overlay, None, False

    best = best.copy()
    best[:, 0, 1] += roi_top
    moments = cv2.moments(best)
    if moments["m00"] == 0:
        return overlay, None, False

    line_x = int(moments["m10"] / moments["m00"])
    line_y = int(moments["m01"] / moments["m00"])
    preferred_half_width = max(82.0, min(width * 0.26, lane_memory["lane_width"] / 2.0))
    side_search_bias = int(width * 0.10)

    if line_x >= center_x:
        lane_x = int(line_x - preferred_half_width - side_search_bias)
        detected_side = "right"
    else:
        lane_x = int(line_x + preferred_half_width + side_search_bias)
        detected_side = "left"

    lane_x = int(clamp(lane_x, 0, width - 1))
    lane_y = line_y
    error = float(lane_x - center_x)

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.rectangle(overlay, (0, roi_top), (width - 1, height - 1), (80, 80, 80), 2)
    cv2.line(overlay, (center_x, roi_top), (center_x, height), (255, 255, 255), 2)
    cv2.drawContours(overlay, [box], 0, (0, 255, 255), 3)
    cv2.circle(overlay, (line_x, line_y), 7, (255, 180, 0), -1)
    cv2.circle(overlay, (lane_x, lane_y), 8, (0, 0, 255), -1)
    cv2.line(overlay, (lane_x, roi_top), (lane_x, height), (0, 220, 120), 2)
    cv2.putText(
        overlay,
        f"Lane offset: {error:.1f}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Single-contour guide: {detected_side}",
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay, error, True


def extract_white_lane_scanline(frame, mask):
    roi_top = int(frame.shape[0] * 0.50)
    roi = mask[roi_top:, :]
    bird_mask, inverse_matrix = build_white_birdeye(roi)
    overlay = frame.copy()
    height, width = frame.shape[:2]
    center_x = width // 2

    cv2.rectangle(overlay, (0, roi_top), (width - 1, height - 1), (80, 80, 80), 2)
    cv2.line(overlay, (center_x, roi_top), (center_x, height), (255, 255, 255), 2)

    valid_centers = []
    left_points = []
    right_points = []
    detected_side = None

    preferred_half_width = max(82.0, min(width * 0.26, lane_memory["lane_width"] / 2.0))
    side_search_bias = int(width * 0.10)
    max_center_gap = int(width * 0.52)
    band_count = 9
    band_height = max(16, bird_mask.shape[0] // band_count)
    global_strength = bird_mask.sum(axis=0).astype(np.float32)
    smooth_kernel = np.ones(19, dtype=np.float32) / 19.0
    global_strength = np.convolve(global_strength, smooth_kernel, mode="same")
    global_threshold = float(255 * max(18, bird_mask.shape[0] * 0.08))
    track_window = int(width * 0.16)

    def find_peak(signal, offset=0, anchor=None):
        if signal.size == 0:
            return -1, 0.0

        work = signal.copy()
        if anchor is not None:
            local_anchor = int(anchor - offset)
            left = max(0, local_anchor - track_window)
            right = min(work.size, local_anchor + track_window)
            masked = np.zeros_like(work)
            masked[left:right] = work[left:right]
            work = masked

        index = int(np.argmax(work))
        value = float(work[index])
        if value < global_threshold:
            return -1, 0.0
        return offset + index, value

    left_anchor, left_anchor_value = find_peak(global_strength[:center_x], 0, None)
    right_anchor, right_anchor_value = find_peak(global_strength[center_x:], center_x, None)
    prev_left = left_anchor if left_anchor_value >= global_threshold else None
    prev_right = right_anchor if right_anchor_value >= global_threshold else None

    for band_index in range(band_count):
        y1 = bird_mask.shape[0] - ((band_index + 1) * band_height)
        y2 = bird_mask.shape[0] - (band_index * band_height)
        y1 = max(0, y1)
        y2 = min(bird_mask.shape[0], y2)
        if y2 - y1 < 8:
            continue

        band = bird_mask[y1:y2, :]
        band_strength = band.sum(axis=0).astype(np.float32)
        band_strength = np.convolve(band_strength, smooth_kernel, mode="same")
        left_peak, left_value = find_peak(band_strength[:center_x], 0, prev_left)
        right_peak, right_value = find_peak(band_strength[center_x:], center_x, prev_right)
        band_y = int((y1 + y2) / 2.0)

        if left_peak >= 0:
            prev_left = left_peak
            left_points.append(map_birdeye_point((left_peak, band_y), inverse_matrix, roi_top))
        if right_peak >= 0:
            prev_right = right_peak
            right_points.append(map_birdeye_point((right_peak, band_y), inverse_matrix, roi_top))

        if left_peak >= 0 and right_peak >= 0:
            gap = right_peak - left_peak
            if 40 <= gap <= max_center_gap:
                lane_memory["lane_width"] = max(80.0, float(gap))
                lane_center = int((left_peak + right_peak) / 2.0)
                valid_centers.append(map_birdeye_point((lane_center, band_y), inverse_matrix, roi_top))
                detected_side = "both"
                continue

        if right_peak >= 0 and (left_peak < 0 or right_value >= left_value * 1.25):
            lane_center = int(right_peak - preferred_half_width - side_search_bias)
            lane_center = int(clamp(lane_center, 0, width - 1))
            valid_centers.append(map_birdeye_point((lane_center, band_y), inverse_matrix, roi_top))
            if detected_side != "both":
                detected_side = "right"
            continue

        if left_peak >= 0:
            lane_center = int(left_peak + preferred_half_width + side_search_bias)
            lane_center = int(clamp(lane_center, 0, width - 1))
            valid_centers.append(map_birdeye_point((lane_center, band_y), inverse_matrix, roi_top))
            if detected_side != "both":
                detected_side = "left"

    if len(valid_centers) < 2:
        fallback_center = None
        if left_anchor >= 0 and right_anchor >= 0:
            gap = right_anchor - left_anchor
            if 40 <= gap <= max_center_gap:
                fallback_center = int((left_anchor + right_anchor) / 2.0)
                lane_memory["lane_width"] = max(80.0, float(gap))
                detected_side = "both"
        elif right_anchor >= 0:
            fallback_center = int(right_anchor - preferred_half_width - side_search_bias)
            detected_side = "right"
        elif left_anchor >= 0:
            fallback_center = int(left_anchor + preferred_half_width + side_search_bias)
            detected_side = "left"

        if fallback_center is None:
            return extract_white_lane_single_contour(frame, mask, roi_top)

        fallback_center = int(clamp(fallback_center, 0, width - 1))
        mid_y = bird_mask.shape[0] // 2
        valid_centers.append(map_birdeye_point((fallback_center, mid_y), inverse_matrix, roi_top))
        valid_centers.append(map_birdeye_point((fallback_center, int(bird_mask.shape[0] * 0.75)), inverse_matrix, roi_top))

    valid_centers.sort(key=lambda point: point[1])
    center_points = np.array(valid_centers, dtype=np.float32)
    guide_points = []

    def fit_single_side(points, side_name):
        if len(points) < 3:
            return None
        ordered = sorted(points, key=lambda point: point[1])
        ys = np.array([point[1] for point in ordered], dtype=np.float32)
        xs = np.array([point[0] for point in ordered], dtype=np.float32)
        fit = np.polyfit(ys, xs, 1)
        lookahead_y = roi_top + int((height - roi_top) * 0.26)
        near_y = roi_top + int((height - roi_top) * 0.88)
        side_lookahead_x = float(np.polyval(fit, lookahead_y))
        side_near_x = float(np.polyval(fit, near_y))

        offset = preferred_half_width + side_search_bias
        if side_name == "left":
            target_lookahead_x = side_lookahead_x + offset
            target_near_x = side_near_x + offset * 0.95
        else:
            target_lookahead_x = side_lookahead_x - offset
            target_near_x = side_near_x - offset * 0.95

        path = []
        for sample_y in np.linspace(lookahead_y, near_y, 6).astype(int):
            side_x = float(np.polyval(fit, sample_y))
            target_x = side_x + offset if side_name == "left" else side_x - offset
            path.append((int(clamp(target_x, 0, width - 1)), int(sample_y)))

        blended_x = int(clamp((target_lookahead_x * 0.78) + (target_near_x * 0.22), 0, width - 1))
        return blended_x, int(lookahead_y), path

    if detected_side == "left":
        fitted = fit_single_side(left_points, "left")
        if fitted is not None:
            lane_x, lane_y, guide_points = fitted
        else:
            fitted = None
    elif detected_side == "right":
        fitted = fit_single_side(right_points, "right")
        if fitted is not None:
            lane_x, lane_y, guide_points = fitted
        else:
            fitted = None
    else:
        fitted = None

    if fitted is None and len(valid_centers) >= 3:
        ys = center_points[:, 1]
        xs = center_points[:, 0]
        fit = np.polyfit(ys, xs, 1)
        lookahead_y = roi_top + int((height - roi_top) * 0.35)
        near_y = roi_top + int((height - roi_top) * 0.78)
        lookahead_x = int(clamp(np.polyval(fit, lookahead_y), 0, width - 1))
        near_x = int(clamp(np.polyval(fit, near_y), 0, width - 1))
        lane_x = int((lookahead_x * 0.72) + (near_x * 0.28))
        lane_y = lookahead_y
        for sample_y in np.linspace(roi_top + 8, height - 8, 6).astype(int):
            sample_x = int(clamp(np.polyval(fit, sample_y), 0, width - 1))
            guide_points.append((sample_x, int(sample_y)))
    elif fitted is None:
        lane_x = int(np.median(center_points[:, 0]))
        lane_y = int(np.median(center_points[:, 1]))

    error = float(lane_x - center_x)

    if left_points and right_points:
        left_x = float(np.median([point[0] for point in left_points]))
        right_x = float(np.median([point[0] for point in right_points]))
        if right_x > left_x:
            lane_memory["lane_width"] = max(80.0, right_x - left_x)

    for point in left_points:
        cv2.circle(overlay, point, 6, (255, 180, 0), -1)
    for point in right_points:
        cv2.circle(overlay, point, 6, (255, 180, 0), -1)
    for point in valid_centers:
        cv2.circle(overlay, point, 5, (0, 220, 120), -1)
    for idx in range(len(guide_points) - 1):
        cv2.line(overlay, guide_points[idx], guide_points[idx + 1], (120, 255, 120), 2)

    cv2.circle(overlay, (lane_x, lane_y), 8, (0, 0, 255), -1)
    cv2.line(overlay, (lane_x, roi_top), (lane_x, height), (0, 220, 120), 2)
    cv2.putText(
        overlay,
        f"Lane offset: {error:.1f}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if detected_side in {"left", "right"}:
        cv2.putText(
            overlay,
            f"Single-side guide: {detected_side}",
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay, error, True


def is_valid_lane_contour(contour, lane_color):
    area = cv2.contourArea(contour)
    if area < 320:
        return False

    if lane_color != "white":
        return True

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    major = max(width, height)
    minor = min(width, height)
    if minor < 1:
        return False

    aspect_ratio = major / minor
    return major >= 85 and aspect_ratio >= 2.8


def extract_lane_from_mask(frame, mask, lane_color):
    roi_top = int(frame.shape[0] * 0.55)
    roi = mask[roi_top:, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, roi_top), (frame.shape[1] - 1, frame.shape[0] - 1), (80, 80, 80), 2)
    center_x = frame.shape[1] // 2
    cv2.line(overlay, (center_x, roi_top), (center_x, frame.shape[0]), (255, 255, 255), 2)

    candidates = []
    for contour in contours:
        if not is_valid_lane_contour(contour, lane_color):
            continue
        area = cv2.contourArea(contour)
        contour[:, 0, 1] += roi_top
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        lane_x = int(moments["m10"] / moments["m00"])
        lane_y = int(moments["m01"] / moments["m00"])
        candidates.append({
            "contour": contour,
            "x": lane_x,
            "y": lane_y,
            "area": area,
        })

    if not candidates:
        return overlay, None, False

    candidates.sort(key=lambda item: item["area"], reverse=True)
    primary = sorted(candidates[:2], key=lambda item: item["x"])

    if len(primary) >= 2:
        left_x = primary[0]["x"]
        right_x = primary[1]["x"]
        lane_x = int((left_x + right_x) / 2.0)
        lane_y = int((primary[0]["y"] + primary[1]["y"]) / 2.0)
        lane_memory["lane_width"] = max(80.0, float(right_x - left_x))
    else:
        only = primary[0]
        estimated_half_width = lane_memory["lane_width"] / 2.0
        lane_x = int(only["x"] + estimated_half_width) if only["x"] < center_x else int(only["x"] - estimated_half_width)
        lane_x = int(clamp(lane_x, 0, frame.shape[1] - 1))
        lane_y = only["y"]

    error = float(lane_x - center_x)

    for item in primary:
        cv2.drawContours(overlay, [item["contour"]], -1, (0, 255, 255), 3)
        cv2.circle(overlay, (item["x"], item["y"]), 7, (255, 180, 0), -1)

    cv2.circle(overlay, (lane_x, lane_y), 8, (0, 0, 255), -1)
    cv2.line(overlay, (lane_x, roi_top), (lane_x, frame.shape[0]), (0, 220, 120), 2)
    cv2.putText(
        overlay,
        f"Lane offset: {error:.1f}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay, error, True


def detect_dark_lane_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_lane(frame, lane_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = build_lane_mask(hsv, lane_color)
    if lane_color == "white":
        overlay, error, detected = extract_white_lane_scanline(frame, color_mask)
    else:
        overlay, error, detected = extract_lane_from_mask(frame, color_mask, lane_color)
    if detected:
        return overlay, error, True

    if lane_color == "white":
        return overlay, error, False

    dark_mask = detect_dark_lane_mask(frame)
    fallback_overlay, fallback_error, fallback_detected = extract_lane_from_mask(frame, dark_mask, "dark")
    if fallback_detected:
        cv2.putText(
            fallback_overlay,
            "Fallback: dark lane",
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return fallback_overlay, fallback_error, True

    return overlay, error, False


def get_recovery_error():
    elapsed = time.time() - lane_memory["last_seen"]
    if elapsed <= 0.7:
        return lane_memory["last_error"], True
    return 0.0, False


def is_red_traffic_light(frame, xyxy):
    x1, y1, x2, y2 = xyxy
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(hsv, np.array((0, 80, 80), dtype=np.uint8), np.array((10, 255, 255), dtype=np.uint8))
    red_mask_2 = cv2.inRange(hsv, np.array((160, 80, 80), dtype=np.uint8), np.array((180, 255, 255), dtype=np.uint8))
    red_ratio = float(np.count_nonzero(red_mask_1 | red_mask_2)) / float(crop.shape[0] * crop.shape[1])
    return red_ratio > 0.08


def run_yolo(frame):
    if model is None:
        return frame, [], False

    try:
        results = model.predict(frame, conf=0.4, verbose=False, imgsz=320)
    except Exception:
        return frame, [], False

    annotated = frame.copy()
    objects = []
    red_light = False

    for result in results:
        annotated = result.plot()
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            objects.append(f"{label} ({conf:.2f})")

            if label.lower() in TRAFFIC_LIGHT_LABELS:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                x1 = clamp(x1, 0, frame.shape[1] - 1)
                x2 = clamp(x2, 0, frame.shape[1] - 1)
                y1 = clamp(y1, 0, frame.shape[0] - 1)
                y2 = clamp(y2, 0, frame.shape[0] - 1)
                if x2 > x1 and y2 > y1 and is_red_traffic_light(frame, (x1, y1, x2, y2)):
                    red_light = True

    return annotated, objects[:6], red_light


def build_ai_message(snapshot):
    if snapshot["mode"] == "manual":
        return f"手動模式，上一個命令：{snapshot['last_command']}"

    if snapshot["red_light"]:
        return "自動駕駛暫停：偵測到紅燈"

    if not snapshot["lane_detected"]:
        return "自動駕駛暫停：未找到車道線"

    if snapshot["objects"]:
        return f"自動循線中，辨識到：{', '.join(snapshot['objects'])}"

    return "自動循線中，前方未偵測到指定物件"


def apply_autonomous_control(error, detected, red_light, speed, dt):
    pwm_speed = speed_to_pwm(speed)

    if not get_state_snapshot()["motor_enabled"]:
        car.stop()
        return "auto blocked: motor unavailable"

    if red_light:
        car.stop()
        return "auto stop: red light"

    if not detected or error is None:
        recovered_error, recoverable = get_recovery_error()
        if not recoverable:
            pid.reset()
            car.stop()
            return "auto stop: lane lost"
        error = recovered_error
        detected = True

    control_error = float(error)
    if abs(control_error) > 40:
        control_error *= 1.45
    elif abs(control_error) > 20:
        control_error *= 1.20

    correction = pid.compute(control_error, dt)
    min_turn = 0.0
    if abs(control_error) > 70:
        min_turn = 12.0
    elif abs(control_error) > 40:
        min_turn = 8.0
    elif abs(control_error) > 22:
        min_turn = 5.0

    if min_turn > 0:
        correction = np.sign(control_error) * max(abs(correction), min_turn)

    correction = clamp(correction, -pwm_speed, pwm_speed)
    car.drive(pwm_speed, correction)
    left_pwm = int(clamp(round(pwm_speed - correction), 0, 100))
    right_pwm = int(clamp(round(pwm_speed + correction), 0, 100))
    return f"auto pwm={pwm_speed} turn={correction:.1f} L={left_pwm} R={right_pwm}"


def processing_loop():
    global stream_jpeg, running

    last_time = time.time()
    yolo_counter = 0
    cached_objects = []
    cached_red_light = False

    while running:
        frame = camera.capture_array()
        if frame is None:
            time.sleep(0.05)
            continue

        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        snapshot = get_state_snapshot()
        lane_overlay, lane_error, lane_detected = detect_lane(frame, snapshot["lane_color"])

        yolo_counter += 1
        annotated = lane_overlay
        if yolo_counter % 3 == 0:
            annotated, cached_objects, cached_red_light = run_yolo(lane_overlay)

        now = time.time()
        dt = now - last_time
        last_time = now

        if snapshot["mode"] == "auto":
            control_debug = apply_autonomous_control(
                lane_error,
                lane_detected,
                cached_red_light,
                snapshot["speed"],
                dt,
            )
        else:
            control_debug = f"manual: {snapshot['last_command']}"

        combined_state = {
            **snapshot,
            "objects": cached_objects,
            "red_light": cached_red_light,
            "lane_detected": lane_detected,
            "lane_offset": 0.0 if lane_error is None else lane_error,
        }
        if lane_detected and lane_error is not None:
            lane_memory["last_error"] = lane_error
            lane_memory["last_seen"] = time.time()
        combined_state["ai_message"] = build_ai_message(combined_state)
        update_state(
            objects=combined_state["objects"],
            red_light=combined_state["red_light"],
            lane_detected=combined_state["lane_detected"],
            lane_offset=combined_state["lane_offset"],
            ai_message=combined_state["ai_message"],
            control_debug=control_debug,
        )

        cv2.putText(
            annotated,
            f"Mode: {combined_state['mode'].upper()}",
            (18, HEIGHT - 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Lane: {combined_state['lane_color']}  Speed: {combined_state['speed']:.2f}",
            (18, HEIGHT - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        ok, buffer = cv2.imencode(".jpg", annotated)
        if ok:
            with frame_lock:
                stream_jpeg = buffer.tobytes()

        time.sleep(0.03)


def gen_frames():
    while True:
        with frame_lock:
            current = stream_jpeg
        if current is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + current + b"\r\n"
        )


def cleanup():
    global running
    running = False
    try:
        car.stop()
    except Exception:
        pass
    try:
        camera.stop()
    except Exception:
        pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(get_state_snapshot())


@app.route("/api/settings", methods=["POST"])
def api_settings():
    payload = request.get_json(silent=True) or request.form

    mode = payload.get("mode", system_state["mode"])
    lane_color = payload.get("lane_color", system_state["lane_color"])
    speed = normalize_speed_value(payload.get("speed", system_state["speed"]))
    kp = float(payload.get("kp", system_state["pid"]["kp"]))
    ki = float(payload.get("ki", system_state["pid"]["ki"]))
    kd = float(payload.get("kd", system_state["pid"]["kd"]))

    if mode not in {"manual", "auto"}:
        return jsonify({"ok": False, "error": "mode 必須是 manual 或 auto"}), 400
    if lane_color not in LANE_COLOR_RANGES:
        return jsonify({"ok": False, "error": "lane_color 必須是 blue/yellow/white"}), 400

    pid.update_gains(kp, ki, kd)

    update_state(
        mode=mode,
        lane_color=lane_color,
        speed=speed,
        pid={"kp": kp, "ki": ki, "kd": kd},
    )

    if mode == "manual":
        car.stop()

    return jsonify({"ok": True, "state": get_state_snapshot()})


@app.route("/control", methods=["POST"])
def control():
    payload = request.get_json(silent=True) or request.form
    action = payload.get("action", "stop")
    snapshot = get_state_snapshot()

    if snapshot["mode"] != "manual":
        return jsonify({"ok": False, "error": "自動模式中無法手動控制"}), 409

    if not snapshot["motor_enabled"]:
        return jsonify({
            "ok": False,
            "error": f"馬達控制初始化失敗: {snapshot['motor_error'] or 'unknown error'}",
        }), 503

    try:
        if action == "stop":
            car.stop()
        else:
            car.move(action, speed=speed_to_pwm(snapshot["speed"]))
        update_state(last_command=action, ai_message=f"手動控制：{action}")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


processing_thread = threading.Thread(target=processing_loop, daemon=True)
processing_thread.start()
atexit.register(cleanup)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    finally:
        cleanup()
