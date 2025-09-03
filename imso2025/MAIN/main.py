#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, math, time, asyncio, threading
import numpy as np
import os

try:
    import websockets  # pip install websockets
except Exception as e:
    websockets = None
    print("[경고] websockets 모듈이 없습니다. pip install websockets 로 설치하세요.")



SHOW_PREVIEW = bool(os.environ.get("DISPLAY"))

# ─────────────────────────────────────────────────────────────
# 설정(필요시 여기 숫자만 바꿔 쓰세요)
# ─────────────────────────────────────────────────────────────
CAM_DEVICE = "/dev/video0"
W, H, FPS = 640, 480, 30           # 카메라 캡처 해상도/프레임
FOV_DEG   = 175.0                  # 광각 카메라 대각/수평 기반 FOV(프로젝트 기준)

# ROI 모드: "margin" | "scale"
ROI_MODE  = "margin"

# margin 모드 파라미터(네 변 절삭)
MARGIN_L, MARGIN_T, MARGIN_R, MARGIN_B = 75, 10, 5, 0

# scale 모드 파라미터(중심 기준 축소/이동)
SHIFT_X, SHIFT_Y, SCALE = 120, 20, 1.0

# 보기용 리사이즈 높이
VIEW_H = 720

# 태양 검출 파라미터(블러/임계/컨투어 스코어)
BLUR_KSIZE = 30
THRESH_VAL = 245
FALLBACK_PERCENTILE = 99.9
MIN_AREA_PIX = 9
CIRC_EXP, RADIAL_K, AR_EXP, AREA_EXP = 5.0, 12.0, 2.0, 0.3

# CUDA 사용 (OpenCV CUDA 빌드 + 디바이스 있을 때만 자동 활성)
USE_CUDA = True

# 화면 좌표 변환(물리 치수/기하 파라미터)
MONITOR_W_CM = 42.4
MONITOR_H_CM = 24.0
L_PARAM_CM   = 15.0   # angles_to_C2()의 l
TILT_DEG     = 50.0   # angles_to_C2()의 t_deg

# ── WebSocket 송신 설정 ──────────────────────────────────────
SEND_TO_WS = True
WS_URL     = "ws://localhost:8000"    # server.py가 띄운 주소(포트 8000) :contentReference[oaicite:4]{index=4}
WS_RETRY_SEC = 2.0                    # 재접속 간격(초)

# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def _has_cuda():
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def draw_text(img, text, org, scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_crosshair(img, pt, size=10, thickness=1, color=(0,255,255)):
    x, y = int(pt[0]), int(pt[1])
    cv2.line(img, (x-size, y), (x+size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y-size), (x, y+size), color, thickness, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# 픽셀 → 각도 → 물리좌표 → 퍼센트 변환(테스트 코드 흐름)
# ─────────────────────────────────────────────────────────────
def pixel_to_angles(dx: int, dy: int):
    """
    중심 기준 픽셀 오프셋 → (θ, φ)[deg]
    dy는 위가 +가 되도록 (h//2 - full_y)로 넣어야 함.
    """
    half = H / 2.0
    dx = max(-half, min(half, dx))
    dy = max(-half, min(half, dy))
    theta = (FOV_DEG/2.0) * (dx / half)
    phi   = (FOV_DEG/2.0) * (dy / half)
    return theta, phi

def angles_to_C2(theta_deg: float, phi_deg: float, l: float = L_PARAM_CM, t_deg: float = TILT_DEG):
    """(θ, φ)[deg] → 2D 좌표 (Cx, Cw) [cm]"""
    t = math.radians(t_deg)
    sin_t, cos_t = math.sin(t), math.cos(t)
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    Cx = l * math.tan(th) * sin_t
    Cw = -l * (math.tan(ph) * sin_t - cos_t)
    return Cx, Cw

def cwcx_to_percent(Cx: float, Cw: float):
    """
    (Cx, Cw)[cm] → (y%, x%)
    - 가로: -W/2..+W/2 → 0..100
    - 세로: 0..H → 0..100
    """
    half_w = MONITOR_W_CM/2.0
    x_pct = (Cx + half_w) / MONITOR_W_CM * 100.0
    y_pct = (Cw / MONITOR_H_CM) * 100.0
    x_pct = max(0, min(100, x_pct))
    y_pct = max(0, min(100, y_pct))
    return int(round(y_pct)), int(round(x_pct))  # (y%, x%)

# ─────────────────────────────────────────────────────────────
# SunDetector (컨투어 기반 ‘원형 밝은 점’ 선택)
# ─────────────────────────────────────────────────────────────
class SunDetector:
    def __init__(self, blur_ksize=BLUR_KSIZE, thresh=THRESH_VAL, use_cuda=True):
        self.blur_ksize = int(blur_ksize) if int(blur_ksize) % 2 == 1 else int(blur_ksize) + 1
        self.thresh = int(thresh)
        self.use_cuda = bool(use_cuda) and _has_cuda()
        self._gfilter = None
        self.prev_r = None
        if self.use_cuda:
            try:
                self._gfilter = cv2.cuda.createGaussianFilter(
                    srcType=cv2.CV_8UC1, dstType=cv2.CV_8UC1,
                    ksize=(self.blur_ksize, self.blur_ksize), sigma1=0
                )
                print(f"[CUDA] enabled. GaussianBlur k={self.blur_ksize}")
            except cv2.error as e:
                print(f"[CUDA] Gaussian filter init failed ({e}). Falling back to CPU.")
                self.use_cuda = False
                self._gfilter = None
        else:
            print("[CUDA] not available. Using CPU path.")

    def _threshold_cpu(self, blur_u8: np.ndarray):
        _, mask = cv2.threshold(blur_u8, self.thresh, 255, cv2.THRESH_BINARY)
        if mask.any():
            return mask
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
        if area <= 0: return None
        peri = cv2.arcLength(cnt, True)
        if peri <= 0: return None
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
            "area": area, "peri": peri, "circularity": circularity, "fill_ratio": fill_ratio,
            "radial_rms": radial_rms, "aspect_ratio": ar,
            "center": (cx, cy), "enclose_center": (int(round(cx_e)), int(round(cy_e))), "radius": r,
        }

    def find_sun(self, bgr: np.ndarray):
        # GRAY + BLUR (CUDA 우선, 실패 시 CPU)
        if self.use_cuda and self._gfilter is not None:
            try:
                g = cv2.cuda_GpuMat(); g.upload(bgr)
                ggray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
                gblur = self._gfilter.apply(ggray)
                blur  = gblur.download()
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

        # 임계 + 모폴로지
        mask = self._threshold_cpu(blur)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        # 컨투어
        contours = self._find_contours(mask)
        if not contours:
            return None, mask, []

        infos = []
        for c in contours:
            m = self._circular_metrics(c)
            if m is None or m["area"] < MIN_AREA_PIX: continue
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
# WebSocket 송신(백그라운드 스레드에서 asyncio 루프 구동: Py3.6 호환)
# ─────────────────────────────────────────────────────────────
class WsSender:
    def __init__(self, url: str, enable: bool = True, retry_sec: float = 2.0):
        self.url = url
        self.enable = enable and (websockets is not None)
        self.retry_sec = retry_sec
        self._stop = threading.Event()
        self._thread = None
        # Py3.6: asyncio.Queue는 루프 필요 → start() 안에서 생성
        self._loop = None
        self._queue = None

    async def _run(self):
        ws = None
        while not self._stop.is_set():
            try:
                if ws is None:
                    ws = await websockets.connect(self.url)
                    print(f"[WS] connected to {self.url}")
                # 타임아웃 폴링
                try:
                    msg = await asyncio.wait_for(self._queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                await ws.send(msg)
            except Exception as e:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                ws = None
                print(f"[WS] disconnected ({e}). retry in {self.retry_sec}s")
                await asyncio.sleep(self.retry_sec)

        if ws is not None:
            try:
                await ws.close()
                print("[WS] closed")
            except Exception:
                pass

    def _runner(self):
        # Py3.6: asyncio.run() 없음 → 전용 루프 구성
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()

    def start(self):
        if not self.enable:
            print("[WS] 비활성 또는 모듈 없음 → 송신 생략")
            return
        self._thread = threading.Thread(target=self._runner, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def send_packet(self, v: int, y: int, x: int):
        if not self.enable or self._loop is None or self._queue is None:
            return
        v = 0 if int(v) == 0 else 1
        y = max(0, min(100, int(y)))
        x = max(0, min(100, int(x)))
        pkt = f"({v},{y},{x})"  # server/script와 동일 포맷 :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
        try:
            # 스레드 안전 큐 삽입 (루프 스레드에서 소비)
            self._loop.call_soon_threadsafe(self._queue.put_nowait, pkt)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# 메인: 캡처 → ROI → 검출 → 각도/퍼센트 변환 → 오버레이/표시 → WS 송신
# ─────────────────────────────────────────────────────────────
def main():
    # WebSocket 송신기 가동
    ws = WsSender(WS_URL, enable=SEND_TO_WS, retry_sec=WS_RETRY_SEC)
    ws.start()

    # 1) GStreamer(MJPEG) 시도
    gst = (
        f"v4l2src device={CAM_DEVICE} ! "
        f"image/jpeg,width={W},height={H},framerate={FPS}/1 ! "
        f"jpegdec ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    # 2) 실패하면 V4L2 + MJPG 강제
    if not cap.isOpened():
        print("[경고] GStreamer 파이프라인 실패 → 기본 V4L2 시도")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS,          FPS)
        # MJPG로 포맷 강제 (해당 카메라 스펙상 30fps 유리) 
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("camera open failed")

    if SHOW_PREVIEW:
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)

    detector = SunDetector(BLUR_KSIZE, THRESH_VAL, use_cuda=(USE_CUDA and _has_cuda()))

    t0, n = time.time(), 0
    last_yx = (50, 50)  # 미검출 시 마지막 좌표 유지용
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # ROI
        if ROI_MODE == "margin":
            x = max(0, min(w - 1, MARGIN_L))
            y = max(0, min(h - 1, MARGIN_T))
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

        if contours:
            cv2.drawContours(roi, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

        sun_text = "Sun: not found"
        if best is not None:
            cx, cy = best["center"]; r = best["radius"]
            cv2.drawContours(roi, [best["contour"]], -1, (0, 255, 0), 2, cv2.LINE_AA)
            if r > 0:
                cv2.circle(roi, best["enclose_center"], int(round(r)), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(roi, (cx, cy), 18, (0, 255, 0), 2, cv2.LINE_AA)
            draw_crosshair(roi, (cx, cy), size=12, thickness=1, color=(0, 255, 0))

            # ROI → 전체 프레임 좌표
            full_x = roi_offset[0] + cx
            full_y = roi_offset[1] + cy

            # 중심 기준 좌표계 (y는 위가 +)
            dx = full_x - (w // 2)
            dy = (h // 2) - full_y

            # 각도 변환 → 2D → 퍼센트
            theta, phi = pixel_to_angles(dx, dy)
            Cx, Cw = angles_to_C2(theta, phi, l=L_PARAM_CM, t_deg=TILT_DEG)
            y_pct, x_pct = cwcx_to_percent(Cx, Cw)
            last_yx = (y_pct, x_pct)

            # 콘솔 로그
            print(f"percent: y={y_pct}%, x={x_pct}%  |  θ={theta:.1f}°, φ={phi:.1f}°  Cx={Cx:.2f}cm Cw={Cw:.2f}cm")

            # 웹소켓 전송 (보임=1)
            ws.send_packet(1, y_pct, x_pct)

            sun_text = (
                f"dx={dx:+d}, dy={dy:+d} | "
                f"theta={theta:.1f}, phi={phi:.1f} | "
                f"Cx={Cx:.2f}cm, Cw={Cw:.2f}cm | "
                f"{x_pct}%, {y_pct}%"
            )
        else:
            # 미검출 시: 원 숨김(0, last_y, last_x)
            ws.send_packet(0, last_yx[0], last_yx[1])

        # 보기용 리사이즈
        disp_w = max(1, int(roi.shape[1] * (VIEW_H / roi.shape[0])))
        disp   = cv2.resize(roi, (disp_w, VIEW_H), interpolation=cv2.INTER_AREA)
        disp_h, disp_w2 = disp.shape[:2]
        draw_crosshair(disp, (disp_w2//2, disp_h//2), size=14, thickness=1, color=(255, 255, 0))

        draw_text(disp, label, (10, 28), scale=0.7)
        draw_text(disp, sun_text, (10, 56), scale=0.7, color=(50, 230, 255))

        # FPS 로그
        n += 1
        if n % 30 == 0:
            dt = time.time() - t0
            if dt > 0:
                print(f"~{n/dt:.1f} FPS @ {int(cap.get(3))}x{int(cap.get(4))}")
            t0, n = time.time(), 0

        if SHOW_PREVIEW:
            cv2.imshow("view", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break
        else:
            # 헤드리스일 경우 잠깐 쉬어주기
            time.sleep(0.001)


    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    ws.stop()

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
