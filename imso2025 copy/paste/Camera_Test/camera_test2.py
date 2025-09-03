#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, sys, time, math
import numpy as np


# === 캡처 기본값 ===
CAM = "/dev/video0"
FOV_DEG = 175.0
W, H, FPS = 640, 480, 30

# === 크롭 모드 선택 ===
MODE = "margin"   # "margin" | "scale"

# --- margin 모드 파라미터 ---
MARGIN_L = 75
MARGIN_T = 10
MARGIN_R = 5
MARGIN_B = 0

# --- scale 모드 파라미터 ---
SHIFT_X = 120
SHIFT_Y = 20
SCALE   = 1.0

# === 표시 크기(보기 성능용) ===
VIEW_H = 720

# === 태양 검출 파라미터 ===
BLUR_KSIZE = 30
THRESH_VAL = 245
FALLBACK_PERCENTILE = 99.9
MIN_AREA_PIX = 9

# === ‘원에 가까움’ 스코어 튜닝 ===
CIRC_EXP = 5.0
RADIAL_K = 12.0
AR_EXP   = 2.0
AREA_EXP = 0.3

# === CUDA 사용 설정 ===
USE_CUDA = False  # True 권장(자동 감지와 AND 연산됨)

def _has_cuda():
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────

def pixel_to_angles(dx: int, dy: int):
    """중심 기준 픽셀 오프셋 -> (θ, φ) [deg].
       y는 위가 +가 되도록 이미 dy = (h//2) - full_y 로 들어온다고 가정."""
    # 클램프
    dx = max(-H/2, min(H/2, dx))
    dy = max(-H/2, min(H/2, dy))
    # 반쪽 비율 * (FOV/2)
    theta = (FOV_DEG/2.0) * (dx / H/2)
    phi   = (FOV_DEG/2.0) * (dy / H/2)
    return theta, phi

def angles_to_C2(theta_deg: float, phi_deg: float, l: float = 15.0, t_deg: float = 50.0):
    """(θ, φ)[deg] -> 2D 좌표 (Cx, Cw)."""
    t = math.radians(t_deg)
    sin_t, cos_t = math.sin(t), math.cos(t)
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    Cx = l * sin_t * math.tan(th)
    Cw = -l * (math.tan(ph) * sin_t - cos_t)
    return Cx, Cw

MONITOR_W_CM = 42.4
MONITOR_H_CM = 24.0

def cwcx_to_percent(Cx: float, Cw: float):
    """
    (Cx, Cw)[cm] → (y%, x%) 변환
    - 가로: -21.2cm ~ +21.2cm → 0~100%
    - 세로: 0 ~ 24cm → 0~100%
    """
    half_w = MONITOR_W_CM/2

    # x: -half_w..+half_w -> 0..100
    x_pct = (Cx + half_w) / MONITOR_W_CM * 100.0

    # y: 0..H -> 0..100
    y_pct = (Cw / MONITOR_H_CM) * 100.0

    # 클램핑
    x_pct = max(0, min(100, x_pct))
    y_pct = max(0, min(100, y_pct))

    return int(round(y_pct)), int(round(x_pct))

def emit_center(x, y, brightness=None):
    """
    표준출력으로 중점 좌표를 내보낸다. (필요시 포맷 바꿔쓰면 됨)
    """
    if brightness is None:
        print(f"CENTER {x} {y}")
    else:
        print(f"CENTER {x} {y} BRIGHT {brightness}")


class SunDetector:
    def __init__(self, blur_ksize=BLUR_KSIZE, thresh=THRESH_VAL, use_cuda=True):
        self.blur_ksize = int(blur_ksize) if int(blur_ksize) % 2 == 1 else int(blur_ksize) + 1
        self.thresh = int(thresh)

        self.use_cuda = bool(use_cuda) and _has_cuda()
        self._gfilter = None
        self.prev_r = None

        if self.use_cuda:
            try:
                # GRAY(8UC1) → GRAY(8UC1) 가우시안 필터
                self._gfilter = cv2.cuda.createGaussianFilter(
                    srcType=cv2.CV_8UC1, dstType=cv2.CV_8UC1,
                    ksize=(self.blur_ksize, self.blur_ksize), sigma1=0
                )
                print(f"[CUDA] enabled. GaussianBlur k={self.blur_ksize}")
            except cv2.error as e:
                # 일부 빌드에서 큰 커널 제한 가능 → CPU로 폴백
                print(f"[CUDA] Gaussian filter init failed ({e}). Falling back to CPU.")
                self.use_cuda = False
                self._gfilter = None
        else:
            print("[CUDA] not available. Using CPU path.")

    def _threshold_cpu(self, blur_u8: np.ndarray):
        # 1차: 고정 임계값
        _, mask = cv2.threshold(blur_u8, self.thresh, 255, cv2.THRESH_BINARY)
        if mask.any():
            return mask
        # 2차: 상위 퍼센타일 백업
        q = np.percentile(blur_u8, FALLBACK_PERCENTILE)
        q = max(q, 200)
        _, mask = cv2.threshold(blur_u8, q, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def _find_contours(mask: np.ndarray):
        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        return contours

    @staticmethod
    def _circular_metrics(cnt):
        area = cv2.contourArea(cnt)
        if area <= 0:
            return None
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            return None

        circularity = 4.0 * math.pi * area / (peri * peri)

        (cx_e, cy_e), r = cv2.minEnclosingCircle(cnt)
        circle_area = math.pi * (r * r) if r > 0 else 1.0
        fill_ratio = float(area) / float(circle_area)

        radial_rms = 1.0
        if r > 1e-6:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            d = np.sqrt((pts[:,0] - cx_e)**2 + (pts[:,1] - cy_e)**2)
            radial_rms = float(np.sqrt(np.mean((d - r)**2)) / r)

        (rx, ry), (rw, rh), ang = cv2.minAreaRect(cnt)
        if rw <= 0 or rh <= 0:
            ar = 0.0
        else:
            short, long = (rw, rh) if rw < rh else (rh, rw)
            ar = float(short / long)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            cx, cy = int(round(cx_e)), int(round(cy_e))
        else:
            cx = int(round(M["m10"]/M["m00"]))
            cy = int(round(M["m01"]/M["m00"]))

        return {
            "area": area,
            "peri": peri,
            "circularity": circularity,
            "fill_ratio": fill_ratio,
            "radial_rms": radial_rms,
            "aspect_ratio": ar,
            "center": (cx, cy),
            "enclose_center": (int(round(cx_e)), int(round(cy_e))),
            "radius": r,
        }

    def find_sun(self, bgr: np.ndarray):
        """가장 '원에 가까운' 컨투어 선택: info(dict), mask, contours 반환 (적응형 블러 포함)"""
        # ---- GRAY + BLUR (CUDA 우선) ----
        if self.use_cuda and self._gfilter is not None:
            try:
                g_bgr = cv2.cuda_GpuMat()
                g_bgr.upload(bgr)  # ROI만 업로드(복사량 최소화)
                g_gray = cv2.cuda.cvtColor(g_bgr, cv2.COLOR_BGR2GRAY)
                g_blur = self._gfilter.apply(g_gray)
                blur   = g_blur.download()  # 이후 단계는 CPU
            except cv2.error:
                # CUDA 런타임 에러 시 CPU 폴백 (적응형 k)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                k = self.blur_ksize
                if self.prev_r is not None:
                    k = int(max(3, min(15, round(self.prev_r * 0.3))))
                    if k % 2 == 0: k += 1
                blur = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            # CPU 경로 (적응형 k)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            k = self.blur_ksize
            if self.prev_r is not None:
                k = int(max(3, min(15, round(self.prev_r * 0.3))))
                if k % 2 == 0: k += 1
            blur = cv2.GaussianBlur(gray, (k, k), 0)

        # ---- THRESHOLD (CPU) ----
        mask = self._threshold_cpu(blur)

        # 연결성 보강(작은 구멍 메우기) - 권장
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        # ---- CONTOURS (CPU) ----
        contours = self._find_contours(mask)
        if not contours:
            return None, mask, []

        infos = []
        for c in contours:
            m = self._circular_metrics(c)
            if m is None or m["area"] < MIN_AREA_PIX:
                continue
            m["contour"] = c
            infos.append(m)
        if not infos:
            return None, mask, contours

        amax = max(i["area"] for i in infos)
        for i in infos:
            circ_term = max(0.0, i["circularity"]) ** CIRC_EXP
            fill_term = max(0.0, min(1.0, i["fill_ratio"])) ** 1.0
            ar_term   = max(0.0, min(1.0, i["aspect_ratio"])) ** AR_EXP
            radial_penalty = math.exp(-RADIAL_K * max(0.0, i["radial_rms"]))
            size_factor = ((i["area"]/amax)**AREA_EXP) if amax > 0 else 1.0
            i["score"] = circ_term * fill_term * ar_term * radial_penalty * size_factor

        best = max(infos, key=lambda d: d["score"])

        # <<< 중요: 여기서만 self.prev_r 업데이트 >>>
        if best is not None:
            self.prev_r = best["radius"]

        return best, mask, contours
# ─────────────────────────────────────────────────────────────
def draw_text(img, text, org, scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_crosshair(img, pt, size=10, thickness=1, color=(0,255,255)):
    x, y = int(pt[0]), int(pt[1])
    cv2.line(img, (x-size, y), (x+size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y-size), (x, y+size), color, thickness, cv2.LINE_AA)

# 카메라 열기 (GStreamer: MJPEG → jpegdec(CPU) → BGR → appsink)
gst = (
    f"v4l2src device={CAM} ! "
    f"image/jpeg,width={W},height={H},framerate={FPS}/1 ! "
    "jpegdec ! videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[경고] GStreamer 파이프라인 실패 → 기본 V4L2 시도")
    cap = cv2.VideoCapture(0)  # fallback

ok, _ = cap.read()
if not ok:
    sys.exit("camera open failed (GStreamer)")

cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)

# CUDA 가능 여부 자동 감지 + 스위치 반영
detector = SunDetector(BLUR_KSIZE, THRESH_VAL, use_cuda=(USE_CUDA and _has_cuda()))

t0, n = time.time(), 0
show_mask = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]

    # ROI 선택
    if MODE == "margin":
        x = max(0, min(w-1, MARGIN_L))
        y = max(0, min(h-1, MARGIN_T))
        cw = max(1, w - x - max(0, MARGIN_R))
        ch = max(1, h - y - max(0, MARGIN_B))
        roi = frame[y:y+ch, x:x+cw]
        label = f"MARGIN roi {cw}x{ch} L{MARGIN_L} R{MARGIN_R} T{MARGIN_T} B{MARGIN_B}"
        roi_offset = (x, y)
    else:
        cw = max(1, min(w, int(w * SCALE)))
        ch = max(1, min(h, int(h * SCALE)))
        x  = max(0, min(w - cw, (w - cw)//2 + SHIFT_X))
        y  = max(0, min(h - ch, (h - ch)//2 + SHIFT_Y))
        roi = frame[y:y+ch, x:x+cw]
        label = f"SCALE roi {cw}x{ch} shift({SHIFT_X},{SHIFT_Y}) scale={SCALE}"
        roi_offset = (x, y)

    # 검출
    best, mask, contours = detector.find_sun(roi)
    if best is not None:
        detector.prev_r = best["radius"]

    # 컨투어 표시(빨강), 선택 컨투어(초록), 최소외접원(노랑)
    if contours:
        cv2.drawContours(roi, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    sun_text = "Sun: not found"
    if best is not None:
        cx, cy = best["center"]
        area   = int(round(best["area"]))
        circ   = best["circularity"]
        fill   = best["fill_ratio"]
        r      = best["radius"]
        peri   = best["peri"]
        rms    = best["radial_rms"]
        ar     = best["aspect_ratio"]

        cv2.drawContours(roi, [best["contour"]], -1, (0, 255, 0), 2, cv2.LINE_AA)
        if r > 0:
            cv2.circle(roi, best["enclose_center"], int(round(r)), (0, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(roi, (cx, cy), 18, (0, 255, 0), 2, cv2.LINE_AA)
        draw_crosshair(roi, (cx, cy), size=12, thickness=1, color=(0, 255, 0))

        full_x = roi_offset[0] + cx
        full_y = roi_offset[1] + cy
        # 기존
        # dx = full_x - (w // 2)
        # dy = (h // 2) - full_y   # 위쪽이 +, 아래쪽이 -

        # 변경: ROI 중심을 원점으로
        cx0 = roi_offset[0] + (cw // 2)   # ROI 중심의 full-frame x
        cy0 = roi_offset[1] + (ch // 2)   # ROI 중심의 full-frame y

        dx = full_x - cx0
        dy = cy0 - full_y  # 위쪽이 +, 아래쪽이 -

        # 클램핑
        dx = max(-240, min(240, dx))
        dy = max(-240, min(240, dy))

        # 각도 변환
        theta, phi = pixel_to_angles(dx, dy)

        # 2D 좌표 변환
        Cx, Cw = angles_to_C2(theta, phi, l=15.0, t_deg=50.0)
        print(f"θ={theta:.1f}°, φ={phi:.1f}° | C=(Cx={Cx:.2f}, Cw={Cw:.2f})")


        # 퍼센트 변환
        y_pct, x_pct = cwcx_to_percent(Cx, Cw)
        print(f" → percent (y={y_pct}%, x={x_pct}%)")

        sun_text = (
            f"dx={dx:+d}, dy={dy:+d} | "
            f"theta={theta:.1f}, pi={phi:.1f} | "
            f"Cx={Cx:.2f}cm, Cw={Cw:.2f}cm | "
            f"{x_pct}%, {y_pct}%"
        )
        emit_center(y_pct, x_pct)

    # 보기용 리사이즈
    disp_w = max(1, int(roi.shape[1] * (VIEW_H / roi.shape[0])))
    disp   = cv2.resize(roi, (disp_w, VIEW_H), interpolation=cv2.INTER_AREA)

    # 프레임 중심 십자
    disp_h, disp_w = disp.shape[:2]
    draw_crosshair(disp, (disp_w//2, disp_h//2), size=14, thickness=1, color=(255, 255, 0))

    # 텍스트
    draw_text(disp, label, (10, 28), scale=0.7)
    draw_text(disp, sun_text, (10, 56), scale=0.7, color=(50, 230, 255))

    # FPS 로그(간단)
    n += 1
    if n % 30 == 0:
        dt = time.time() - t0
        if dt > 0:
            print(f"~{n/dt:.1f} FPS @ {int(cap.get(3))}x{int(cap.get(4))}")
        t0, n = time.time(), 0

    cv2.imshow("view", disp)

    # 마스크 보기 토글
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('m'):
        mdisp = cv2.resize(mask, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("mask", mdisp)

cap.release()
cv2.destroyAllWindows()
