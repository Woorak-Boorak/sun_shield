#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import argparse
import os
from collections import deque

# ===== 기본 설정 =====
CAM_INDEX   = 0
FRAME_W     = 1280
FRAME_H     = 720
FPS         = 30

# 노출 2단계 (주의: 드라이버가 적용 안정화에 2~5프레임 필요)
EXPO_LOW    = 50    # 야외(직사광) 기준 시작점: 20~80
EXPO_NORM   = 120   # 실내 기준 시작점: 80~200

# 후보 스코어 가중치
W_LOCAL_CONTRAST = 0.6
W_ROUNDNESS      = 0.25
W_SAT_PENALTY    = 0.15

# 임계값/필터
V_THR_NORM = 220
V_THR_LOW  = 200
S_MAX      = 40
MIN_AREA   = 4

# 지속성
PERSIST_N  = 5
PERSIST_OK = 3
history    = deque(maxlen=PERSIST_N)

def open_cam(dev_index, w, h, fps):
    # V4L2 백엔드로 열기
    cap = cv2.VideoCapture(dev_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    # MJPG 강제
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # 색 변환 허용
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except:
        pass

    # 워밍업(버퍼 채우기)
    t0 = time.time()
    ok, frm = False, None
    while time.time() - t0 < 1.0:
        ok, frm = cap.read()
        if ok and frm is not None and frm.size > 0:
            break
    if not ok:
        print("[ERR] Camera warmup failed")
    return cap

def set_manual_exposure(cap, expo_abs):
    """일부 드라이버는 CAP_PROP_AUTO_EXPOSURE 값이 0.25(수동), 0.75(자동). 둘 다 시도."""
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except:
        pass
    # 몇 드라이버는 1이 수동이기도 함 → 백업 시도
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except:
        pass
    # 노출 설정
    cap.set(cv2.CAP_PROP_EXPOSURE, float(expo_abs))

def grab_after_set(cap, settle_frames=4):
    """설정 직후 안정화용 프레임 버림 후 1프레임 반환"""
    ok = False
    frame = None
    for _ in range(settle_frames):
        cap.read()
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.01)
    return ok, frame

def find_contours_compat(binary_img):
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res
    return contours

def score_candidate(gray, hsv, mask, cx, cy):
    h, w = gray.shape
    r = 15
    x0 = max(0, cx - r); x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r); y1 = min(h, cy + r + 1)

    patch = gray[y0:y1, x0:x1]
    local_max = float(np.max(patch))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    cy_abs = cy; cx_abs = cx
    rr = (yy - cy_abs)**2 + (xx - cx_abs)**2
    ring = patch[(rr >= 6**2) & (rr <= 15**2)]
    if ring.size < 10:
        ring_p95 = np.percentile(patch, 95)
    else:
        ring_p95 = np.percentile(ring, 95)
    local_contrast = (local_max + 1.0) / (ring_p95 + 1.0)

    s_patch = hsv[y0:y1, x0:x1, 1]
    s_med = float(np.median(s_patch))
    sat_bonus = 1.0 if s_med < S_MAX else 0.4

    ys, xs = np.where(mask > 0)
    if len(xs) >= 5:
        pts = np.vstack([xs, ys]).astype(np.float32).T
        cov = np.cov(pts, rowvar=False)
        evals, _ = np.linalg.eig(cov)
        evals = np.sort(evals)
        roundness = float(evals[0] / (evals[1] + 1e-6))
    else:
        roundness = 0.3

    score = (W_LOCAL_CONTRAST * local_contrast) + \
            (W_ROUNDNESS * roundness) + \
            (W_SAT_PENALTY * sat_bonus)
    return score

def detect_light(frame_low, frame_norm):
    blur = 3
    g_low  = cv2.GaussianBlur(cv2.cvtColor(frame_low,  cv2.COLOR_BGR2GRAY), (blur,blur), 0)
    g_norm = cv2.GaussianBlur(cv2.cvtColor(frame_norm, cv2.COLOR_BGR2GRAY), (blur,blur), 0)

    _, b1 = cv2.threshold(g_norm, V_THR_NORM, 255, cv2.THRESH_BINARY)
    _, b2 = cv2.threshold(g_low,  V_THR_LOW,  255, cv2.THRESH_BINARY)
    core = cv2.bitwise_and(b1, b2)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    sat_mask = cv2.inRange(s, 0, S_MAX)
    core = cv2.bitwise_and(core, sat_mask)

    contours = find_contours_compat(core)
    if not contours:
        return None, core

    best_score = None
    best_info = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])

        cand_mask = np.zeros_like(core)
        cv2.drawContours(cand_mask, [cnt], -1, 255, -1)
        sc = score_candidate(g_norm, hsv, cand_mask, cx, cy)
        if best_score is None or sc > best_score:
            best_score = sc
            best_info = (cx, cy, sc)
    return best_info, core

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", type=int, default=0)
    parser.add_argument("--device", type=int, default=CAM_INDEX)
    parser.add_argument("--w", type=int, default=FRAME_W)
    parser.add_argument("--h", type=int, default=FRAME_H)
    args = parser.parse_args()

    use_gui = (args.nogui == 0) and bool(os.environ.get("DISPLAY"))
    cap = open_cam(args.device, args.w, args.h, FPS)

    if not cap or not cap.isOpened():
        print("[ERR] Camera open failed")
        return

    print("[INFO] q: quit, d: toggle debug")
    show_debug = True
    last_decision = None

    # 첫 프레임에서 평균 밝기가 비정상(너무 어둡거나 0)이면 안내 문구 출력
    def overlay_info(img, text, y=30, color=(0,0,255)):
        cv2.putText(img, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    while True:
        # --- 낮은 노출 캡쳐 ---
        set_manual_exposure(cap, EXPO_LOW)
        ok1, f_low = grab_after_set(cap, settle_frames=5)

        # --- 보통 노출 캡쳐 ---
        set_manual_exposure(cap, EXPO_NORM)
        ok2, f_norm = grab_after_set(cap, settle_frames=5)

        if not (ok1 and ok2):
            # 프레임 실패 시, 검정 캔버스에 경고 출력
            h = FRAME_H; w = FRAME_W
            out = np.zeros((h, w, 3), dtype=np.uint8)
            overlay_info(out, "No frames from camera. Check MJPG/format/exposure.", 30)
            if use_gui:
                cv2.imshow("Light Detect (norm|core)", out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            else:
                print("[WARN] frame grab failed")
            continue

        # 평균 밝기 점검(완전 암전 방지)
        mean_low  = float(np.mean(cv2.cvtColor(f_low,  cv2.COLOR_BGR2GRAY)))
        mean_norm = float(np.mean(cv2.cvtColor(f_norm, cv2.COLOR_BGR2GRAY)))

        info, core = detect_light(f_low, f_norm)
        if info is not None:
            history.append((info[0], info[1]))
            xs = np.array([p[0] for p in history]); ys = np.array([p[1] for p in history])
            if len(xs) >= PERSIST_OK:
                cx_med = int(np.median(xs)); cy_med = int(np.median(ys))
                close = np.sum((np.hypot(xs - cx_med, ys - cy_med) < 15).astype(np.int32))
                if close >= PERSIST_OK:
                    last_decision = (cx_med, cy_med, info[2])
        else:
            history.clear()

        out = f_norm.copy()
        if last_decision is not None:
            cx, cy, sc = last_decision
            cv2.circle(out, (cx, cy), 12, (0,0,255), 2)
            overlay_info(out, "LIGHT ({}, {}) score {:.2f}".format(cx, cy, sc), 30)
        else:
            overlay_info(out, "No light detected", 30, (0,255,255))

        overlay_info(out, "Mean(L/N) = {:.1f}/{:.1f} | Expo(L/N) = {}/{}".format(
            mean_low, mean_norm, EXPO_LOW, EXPO_NORM), 60, (255,255,0))

        core_viz = cv2.cvtColor(core, cv2.COLOR_GRAY2BGR) if core is not None else np.zeros_like(out)
        vis = np.hstack([out, core_viz])

        if use_gui:
            cv2.imshow("Light Detect (norm|core)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            if last_decision is not None:
                cx, cy, sc = last_decision
                print("LIGHT {},{} score {:.2f}".format(cx, cy, sc))

    cap.release()
    if use_gui:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
