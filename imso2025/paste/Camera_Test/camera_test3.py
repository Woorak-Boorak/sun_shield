#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, math, numpy as np

def _draw_text(img, text, org, scale=0.7, color=(255,255,255), thickness=2):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def _draw_crosshair(img, pt, size=10, thickness=1, color=(0,255,255)):
    import cv2
    x, y = int(pt[0]), int(pt[1])
    cv2.line(img, (x-size, y), (x+size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y-size), (x, y+size), color, thickness, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# [Sun Detect 관련 시작]────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

# =========================
# [전역 상태] 초기화/종료에서 세팅하고, 읽기에서 재사용
# =========================
_cap = None            # cv2.VideoCapture 핸들 (sun_init에서 생성, sun_close에서 해제)
_detector = None       # SunDetector 인스턴스 (블러/임계 등 내부 상태 포함)
_roi_mode = "margin"   # "margin" 또는 "scale"
_roi_offset = (0, 0)   # ROI의 원본 프레임 내 시작 좌표 (x, y)
_roi_size = (0, 0)     # ROI 크기 (cw, ch)

# =========================
# [설정] 카메라/ROI/임계값
# =========================
CAM_DEVICE = "/dev/video0"
W, H, FPS = 640, 480, 30

# margin 모드: 네 변에서 잘라내기
MARGIN_L, MARGIN_T, MARGIN_R, MARGIN_B = 75, 10, 5, 0

# scale 모드: 중심 기준 축소/이동
SHIFT_X, SHIFT_Y, SCALE = 120, 20, 1.0

# 임계/블러 파라미터
BLUR_KSIZE = 30               # 가우시안 블러 커널(홀수)
THRESH_VAL = 245              # 1차 고정 임계값
FALLBACK_PERCENTILE = 99.9    # 실패 시 상위 퍼센타일 임계값
MIN_AREA_PIX = 9              # 너무 작은 잡음 제거

# '원형성' 스코어 가중치
CIRC_EXP = 5.0
RADIAL_K = 12.0
AR_EXP   = 2.0
AREA_EXP = 0.3

USE_CUDA = True               # OpenCV CUDA 빌드 + 디바이스 있으면 자동 사용


# 보기용(필요시 조정)
_VIEW_H = 720
_show_mask = False
_last_fps_t = None
_last_fps_n = 0
# ─────────────────────────────────────────────────────────────

def _has_cuda():
    """CUDA 사용 가능 여부 확인(OpenCV CUDA 빌드 + 디바이스 존재)."""
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


class SunDetector:
    """
    입력 BGR 이미지를 받아 '밝고 둥근' 후보를 컨투어 기반으로 평가/선정.
    - 그레이 변환 → 블러 → 임계 이진화 → 모폴로지 → 컨투어 → 지표 계산/스코어링
    - prev_r(이전 프레임 반지름)에 따라 블러 커널을 적응형 조정
    """
    def __init__(self, blur_ksize=BLUR_KSIZE, thresh=THRESH_VAL, use_cuda=True):
        # 블러 커널은 홀수만 허용 → 짝수면 +1
        self.blur_ksize = int(blur_ksize) if int(blur_ksize) % 2 == 1 else int(blur_ksize) + 1
        self.thresh = int(thresh)
        self.use_cuda = bool(use_cuda) and _has_cuda()
        self._gfilter = None
        self.prev_r = None  # 이전 프레임 반지름(적응형 블러용)

        if self.use_cuda:
            try:
                # GRAY(8UC1) → GRAY(8UC1) 가우시안 필터
                self._gfilter = cv2.cuda.createGaussianFilter(
                    srcType=cv2.CV_8UC1, dstType=cv2.CV_8UC1,
                    ksize=(self.blur_ksize, self.blur_ksize), sigma1=0
                )
                print(f"[CUDA] enabled. GaussianBlur k={self.blur_ksize}")
            except cv2.error as e:
                # 일부 빌드/디바이스에서 큰 커널 제한 → CPU 폴백
                print(f"[CUDA] Gaussian filter init failed ({e}). Falling back to CPU.")
                self.use_cuda = False
                self._gfilter = None
        else:
            print("[CUDA] not available. Using CPU path.")

    def _threshold_cpu(self, blur_u8: np.ndarray):
        """
        임계 이진화:
        1) 고정 임계값(THRESH_VAL) 시도
        2) 마스크가 비면 상위 퍼센타일(FALLBACK_PERCENTILE) 값을 임계로 폴백
        """
        _, mask = cv2.threshold(blur_u8, self.thresh, 255, cv2.THRESH_BINARY)
        if mask.any():
            return mask
        q = np.percentile(blur_u8, FALLBACK_PERCENTILE)
        q = max(q, 200)  # 너무 낮은 임계 방지
        _, mask = cv2.threshold(blur_u8, q, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def _find_contours(mask: np.ndarray):
        """OpenCV 3/4 호환 컨투어 추출."""
        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        return contours

    @staticmethod
    def _circular_metrics(cnt):
        """
        컨투어의 원형성 지표 계산:
        - area, peri(둘레), circularity, fill_ratio(최소외접원 대비 채움비),
          radial_rms(반경 오차 정규화), aspect_ratio(최소 외접 직사각형 종횡비),
          center(무게중심), enclose_center/radius(최소외접원)
        """
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
            d = np.sqrt((pts[:, 0] - cx_e)**2 + (pts[:, 1] - cy_e)**2)
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
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))

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
        """
        BGR ROI에서 가장 '원에 가까운' 밝은 영역 1개를 선정.
        반환: (best_info(dict) 또는 None, mask, contours)
        """
        # --- GRAY + BLUR (CUDA 우선, 실패 시 CPU) ---
        if self.use_cuda and self._gfilter is not None:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(bgr)
                ggray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
                gblur = self._gfilter.apply(ggray)
                blur = gblur.download()
            except cv2.error:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                k = self.blur_ksize
                if self.prev_r is not None:
                    k = int(max(3, min(15, round(self.prev_r * 0.3))))
                    if k % 2 == 0: k += 1
                blur = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            k = self.blur_ksize
            if self.prev_r is not None:
                k = int(max(3, min(15, round(self.prev_r * 0.3))))
                if k % 2 == 0: k += 1
            blur = cv2.GaussianBlur(gray, (k, k), 0)

        # --- 임계 이진화 + 모폴로지 ---
        mask = self._threshold_cpu(blur)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        # --- 컨투어 → 지표/스코어링 ---
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
            size_factor = ((i["area"] / amax) ** AREA_EXP) if amax > 0 else 1.0
            i["score"] = circ_term * fill_term * ar_term * radial_penalty * size_factor

        best = max(infos, key=lambda d: d["score"])
        if best is not None:
            self.prev_r = best["radius"]
        return best, mask, contours

# ─────────────────────────────────────────────────────────────
# 1) 초기화: 카메라 열고, ROI 정의, 검출기 준비
# ─────────────────────────────────────────────────────────────
def sun_init(roi_mode="margin"):
    """
    카메라를 열고 검출기를 준비한다. (반드시 프로그램에서 1번만 호출)
    - roi_mode: "margin" 또는 "scale"
    """
    global _cap, _detector, _roi_mode, _roi_offset, _roi_size

    _roi_mode = roi_mode

    # GStreamer: MJPEG → jpegdec → BGR → appsink (Jetson에서 빠름)
    gst = (
        f"v4l2src device={CAM_DEVICE} ! "
        f"image/jpeg,width={W},height={H},framerate={FPS}/1 ! "
        f"jpegdec ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[WARN] GStreamer open failed → fallback to V4L2 index 0")
        cap = cv2.VideoCapture(0)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("camera open/read failed")

    # ROI 미리 계산 (매 프레임 동일하게 사용)
    h, w = frame.shape[:2]
    if _roi_mode == "margin":
        x = max(0, min(w - 1, MARGIN_L))
        y = max(0, min(h - 1, MARGIN_T))
        cw = max(1, w - x - max(0, MARGIN_R))
        ch = max(1, h - y - max(0, MARGIN_B))
    else:
        cw = max(1, min(w, int(w * SCALE)))
        ch = max(1, min(h, int(h * SCALE)))
        x = max(0, min(w - cw, (w - cw) // 2 + SHIFT_X))
        y = max(0, min(h - ch, (h - ch) // 2 + SHIFT_Y))

    _roi_offset = (x, y)
    _roi_size = (cw, ch)

    _cap = cap
    _detector = SunDetector(blur_ksize=BLUR_KSIZE, thresh=THRESH_VAL, use_cuda=USE_CUDA)
    print(f"[INIT] camera {W}x{H}@{FPS}, ROI mode={_roi_mode}, offset={_roi_offset}, size={_roi_size}")


# ─────────────────────────────────────────────────────────────
# 2) 읽기: 프레임 1장을 처리하고 (x, y, brightness) 반환
# ─────────────────────────────────────────────────────────────
def sun_read():
    """
    한 프레임만 읽어 태양 후보를 검출하고,
    원본 프레임 좌표계 기준 (x, y, brightness)을 반환한다.
    - 실패 시: (None, None, None)
    - 화면 표시/GUI 없음. 진단은 필요 시 여기서 print 추가 가능.
    """
    global _cap, _detector, _roi_offset, _roi_size

    if _cap is None or _detector is None:
        raise RuntimeError("sun_init()을 먼저 호출하세요.")

    ok, frame = _cap.read()
    if not ok:
        print("[WARN] frame grab failed")
        return (None, None, None)

    # ROI 자르기 (초기화 때 미리 계산한 offset/size 사용)
    x0, y0 = _roi_offset
    cw, ch = _roi_size
    roi = frame[y0:y0 + ch, x0:x0 + cw]

    # 검출 수행
    best, mask, contours = _detector.find_sun(roi)
    if best is None:
        return (None, None, None)

    # 좌표: ROI → 원본 프레임으로 변환
    cx_roi, cy_roi = best["center"]
    cx_full = int(x0 + cx_roi)
    cy_full = int(y0 + cy_roi)

    # 밝기: 선택 영역(컨투어) 내부의 원본 GRAY 최대값
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sun_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(sun_mask_roi, [best["contour"]], -1, color=255, thickness=-1)

    sun_mask_full = np.zeros(gray_full.shape[:2], dtype=np.uint8)
    sun_mask_full[y0:y0 + ch, x0:x0 + cw] = sun_mask_roi

    if np.any(sun_mask_full):
        brightness = int(gray_full[sun_mask_full > 0].max())
    else:
        brightness = int(gray_full[cy_full, cx_full])

    # (필요시) 여기서 디버그 로그를 출력해도 됨:
    # print(f"[Sun] ({cx_full},{cy_full}) brightness={brightness} r~{best['radius']:.1f}")

    return (cx_full, cy_full, brightness)

def sun_read_vis(return_image=False, window_name="SunView"):
    """
    한 프레임을 읽고, ROI 내 검출 결과를 오버레이해서 즉시 화면에 띄운다.
    - 키보드: ESC 종료(False 반환), 'm' 마스크 토글
    - return_image=True면 (BGR 시각화 프레임)도 함께 반환
    반환:
      cont, x, y, brightness           (return_image=False)
      cont, x, y, brightness, vis_bgr  (return_image=True)
    """
    import cv2, time
    global _cap, _detector, _roi_offset, _roi_size
    global _show_mask, _last_fps_t, _last_fps_n

    if _cap is None or _detector is None:
        raise RuntimeError("sun_init()을 먼저 호출하세요.")

    ok, frame = _cap.read()
    if not ok:
        print("[WARN] frame grab failed")
        return False, None, None, None if not return_image else (False, None, None, None, None)

    x0, y0 = _roi_offset
    cw, ch = _roi_size
    roi = frame[y0:y0 + ch, x0:x0 + cw].copy()
    h, w = frame.shape[:2]

    # 검출
    best, mask, contours = _detector.find_sun(roi)

    # 컨투어/원 표시
    if contours:
        cv2.drawContours(roi, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    sun_text = "Sun: not found"
    full_x = full_y = brightness = None

    if best is not None:
        cx, cy = best["center"]
        r      = best["radius"]
        cv2.drawContours(roi, [best["contour"]], -1, (0, 255, 0), 2, cv2.LINE_AA)
        if r > 0:
            cv2.circle(roi, best["enclose_center"], int(round(r)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(roi, (cx, cy), 18, (0, 255, 0), 2, cv2.LINE_AA)
        _draw_crosshair(roi, (cx, cy), size=12, thickness=1, color=(0, 255, 0))

        full_x = x0 + cx
        full_y = y0 + cy
        dx = full_x - (w // 2)
        dy = full_y - (h // 2)

        # 밝기 측정(컨투어 내부 최대값)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sun_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(sun_mask_roi, [best["contour"]], -1, 255, -1)
        sun_mask_full = np.zeros(gray_full.shape[:2], dtype=np.uint8)
        sun_mask_full[y0:y0 + ch, x0:x0 + cw] = sun_mask_roi
        if np.any(sun_mask_full):
            brightness = int(gray_full[sun_mask_full > 0].max())
        else:
            brightness = int(gray_full[full_y, full_x])

        sun_text = f"Sun full({full_x},{full_y}) roi({cx},{cy}) d_center({dx:+d},{dy:+d}) r~{r:.1f}"

    # 보기용 리사이즈
    disp_w = max(1, int(roi.shape[1] * (_VIEW_H / roi.shape[0])))
    disp = cv2.resize(roi, (disp_w, _VIEW_H), interpolation=cv2.INTER_AREA)

    # 프레임 중심 십자(보기 기준)
    dh, dw = disp.shape[:2]
    _draw_crosshair(disp, (dw//2, dh//2), size=14, thickness=1, color=(255, 255, 0))

    # 라벨/텍스트
    label = f"{_roi_mode.upper()} roi {cw}x{ch}"
    _draw_text(disp, label, (10, 28), scale=0.7)
    _draw_text(disp, sun_text, (10, 56), scale=0.7, color=(50, 230, 255))

    # FPS 로그(30프레임마다)
    _last_fps_n = (_last_fps_n or 0) + 1
    _last_fps_t = _last_fps_t or time.time()
    if _last_fps_n % 30 == 0:
        dt = time.time() - _last_fps_t
        if dt > 0:
            print(f"~{_last_fps_n/dt:.1f} FPS @ {int(_cap.get(3))}x{int(_cap.get(4))}")
        _last_fps_t, _last_fps_n = time.time(), 0

    # 화면 표시
    cv2.imshow(window_name, disp)

    # 마스크 보기 토글
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        return (False, full_x, full_y, brightness) if not return_image else (False, full_x, full_y, brightness, disp)
    elif key == ord('m') and mask is not None:
        mdisp = cv2.resize(mask, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"{window_name}-mask", mdisp)

    return (True, full_x, full_y, brightness) if not return_image else (True, full_x, full_y, brightness, disp)

# ─────────────────────────────────────────────────────────────
# 3) 종료: 자원 해제
# ─────────────────────────────────────────────────────────────
def sun_close():
    """카메라 자원을 해제한다. (프로그램 종료 시 1번 호출)"""
    global _cap, _detector
    try:
        if _cap is not None:
            _cap.release()
    finally:
        _cap = None
        _detector = None
        print("[CLOSE] camera released")

# ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────[Sun Detect 관련 끝]
# ─────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 초기화 (한 번만)
    sun_init(roi_mode="margin")  # 또는 "scale"

    try:
        # 2) 반복 사용 (루프에서 sun_read만 계속 호출)
        for _ in range(300):   # 예: 300프레임 읽기
            x, y, val = sun_read()
            print("Sun:", x, y, val)
            # 여기서 (x, y, val)을 이용해 다음 단계 로직(원 그리기/매핑 등) 수행
            # time.sleep(0.01)  # 필요하면 템포 조절
    finally:
        # 3) 종료 (한 번만)
        sun_close()
