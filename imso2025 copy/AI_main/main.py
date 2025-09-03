#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
Jetson Nano (Ubuntu 18.04 / OpenCV 3.x)에서 두 개의 USB 카메라를 사용해
- 광각 카메라로 '햇빛(강한 하이라이트)'의 위치를 추정하고
- 내부 카메라로 '사람 눈'의 위치를 추정한 뒤
두 위치 관계를 바탕으로 흰색 캔버스 위에 회색~검정 원을 표시하는 데모.

구성 요소
- functions.py: 장치 제어/유틸 (카메라 열기, 노출 설정, 텍스트/캔버스, Haar 로드 등)
- core.py     : 시각 알고리즘 (햇빛 검출, 눈 검출, 원 위치/강도 결정, 원 그리기)

실행 모드
- GUI 모드  : DISPLAY가 설정되어 있고 --nogui 0(기본)일 때. 창으로 결과 표시.
- Headless  : DISPLAY가 없거나 --nogui 1일 때. 표준출력으로 로그만 표시.

중요 팁
- GUI가 필요하면 Jetson에서 X 세션이 떠 있어야 하며, 아래 환경 변수를 셋업:
  $ DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority python main.py
- 광원 검출은 저노출 프레임에서 더 안정적이라, 루프당 일정 주기(10프레임)에만
  노출을 낮춰 한 번 캡처(f_low_cached)하고 나머지는 보통 노출로 시청/표시.
"""

import cv2
import numpy as np
import argparse
import os

# ---------- 외부 모듈 (동일 디렉토리 내 파일) ----------
from functions import (
    open_cam,            # 카메라 열기 (해상도/FPS 설정 포함)
    set_manual_exposure, # 수동 노출로 전환 및 노출값 설정
    grab_after_set,      # 설정 직후 안정화 프레임을 버리고 유효 프레임 획득
    overlay_text,        # 캔버스에 디버그 텍스트 표시
    make_white_canvas,   # 출력용 흰색 캔버스 생성
    load_eye_cascade     # 눈 검출용 Haar Cascade 로더
)
from core import (
    detect_sunlight,     # 저노출/보통노출 프레임을 비교해 강한 하이라이트(햇빛) 위치 추정
    detect_eye,          # 내부 카메라에서 눈 위치 (x, y) 검출
    decide_circle,       # 햇빛/눈 좌표를 바탕으로 최종 원의 중심과 강도(intensity) 결정
    draw_circle          # 흰 캔버스에 원을 실제로 그림
)

# ===== 기본 설정 =====
WIDE_CAM   = 0       # 광각(바깥) 카메라: 보통 /dev/video0
INSIDE_CAM = 1       # 내부(눈)   카메라: 보통 /dev/video1 (미연결 시 자동 스킵)
W, H, FPS  = 1280, 720, 30

# 노출(광원 검출용)
# - EXPO_LOW : 하이라이트(햇빛) 포화 억제를 위해 낮은 노출(야외 20~80 권장)
# - EXPO_NORM: 시청/표시용 보통 노출(실내 80~200 권장)
EXPO_LOW  = 50
EXPO_NORM = 120


def main():
    """
    프로그램의 엔트리 포인트.
    1) 인자 파싱
    2) GUI 가능 여부 판단
    3) 두 카메라를 시도해 열고(내부 카메라는 실패 시 비활성화)
    4) 눈 검출용 Haar 로드 (실패 시 내부 카메라 비활성화)
    5) 루프:
       - 저노출 프레임은 10프레임에 1번만 새로 캡처(속도/안정성 목적)
       - 보통 노출 프레임은 매 루프 캡처(시청/표시 목적)
       - 저노출/보통노출 조합으로 햇빛 좌표 추정
       - 내부 카메라가 있으면 눈 좌표 추정
       - decide_circle로 원의 중심/강도 산출 후 draw_circle로 그리기
       - GUI면 창에 표시, Headless면 로그 출력
    6) 종료 정리(release/destroy)
    """
    # ----- 1) 인자 파싱 -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui",  type=int, default=0, help="1이면 Headless(창 미표시)")
    parser.add_argument("--wide",   type=int, default=WIDE_CAM, help="광각 카메라 인덱스")
    parser.add_argument("--inside", type=int, default=INSIDE_CAM, help="내부 카메라 인덱스")
    parser.add_argument("--w",      type=int, default=W, help="입력 해상도 가로")
    parser.add_argument("--h",      type=int, default=H, help="입력 해상도 세로")
    args = parser.parse_args()

    # ----- 2) GUI 가능 여부 판단 -----
    # - DISPLAY 환경변수가 없으면 창을 띄울 수 없음 (Gtk-WARNING 방지)
    # - --nogui 1이면 강제로 Headless
    use_gui = (args.nogui == 0) and bool(os.environ.get("DISPLAY"))
    if not use_gui:
        print("[INFO] Headless mode (no DISPLAY). "
              "GUI가 필요하면 DISPLAY=:0 및 XAUTHORITY를 올바르게 설정하세요.")

    # ----- 3) 카메라 열기 -----
    # 광각 카메라(필수): 실패하면 프로그램 종료
    cam_wide = open_cam(args.wide, args.w, args.h, FPS)
    if (not cam_wide) or (not cam_wide.isOpened()):
        print("[ERR] Wide camera open failed")
        return

    # 내부 카메라(선택): 실패하면 눈 검출 비활성화
    cam_inside = open_cam(args.inside, args.w, args.h, FPS)
    if (not cam_inside) or (not cam_inside.isOpened()):
        print("[WARN] Inside camera not found. Eye detection disabled.")
        cam_inside = None

    # ----- 4) 눈 검출용 Cascade 로드 -----
    # - 내부 카메라가 열려 있고 haarcascade_eye.xml을 찾지 못하면 눈 검출을 끕니다.
    eye_cascade = load_eye_cascade()
    if eye_cascade is None and cam_inside is not None:
        print("[WARN] haarcascade_eye.xml not found. Eye detection disabled.")
        cam_inside = None

    print("[INFO] q: quit (GUI 모드에서 창 활성 상태일 때)")

    # 루프 상태 변수
    frame_count = 0
    f_low_cached = None  # 저노출 프레임 캐시(10프레임에 1회 갱신)

    try:
        while True:
            frame_count += 1

            # ----- 5-1) 저노출 프레임 준비 (10프레임마다 1번만) -----
            # 노출 전환은 드라이버 안정화 시간이 필요하므로 너무 자주 바꾸면 FPS 하락/깜빡임 발생.
            # settle_frames=2로 몇 프레임을 버리고 안정된 프레임을 수집.
            if (frame_count % 10) == 1:
                set_manual_exposure(cam_wide, EXPO_LOW)
                ok_low, f_low = grab_after_set(cam_wide, settle_frames=2)
                f_low_cached = f_low if ok_low else None  # 실패 시 None 유지

            # ----- 5-2) 보통 노출 프레임 (매 루프) -----
            # 시청/표시용으로 항상 보통 노출 프레임을 최신 상태로 갱신.
            set_manual_exposure(cam_wide, EXPO_NORM)
            ok_norm, f_norm = grab_after_set(cam_wide, settle_frames=2)

            # ----- 5-3) 햇빛 좌표 추정 -----
            # detect_sunlight는 저노출/보통노출 프레임을 비교하여
            # '보통에서는 밝고, 저노출에서도 여전히 매우 밝은' 영역(하이라이트)을 찾아냄.
            # 파라미터 설명:
            # - v_thr_norm: 보통 노출 프레임에서 밝기(V)가 이 값 이상인 픽셀만 후보
            # - v_thr_low : 저노출 프레임에서도 밝기가 이 값 이상이면 '진짜 강한 광원'으로 신뢰 상승
            # - s_max     : 채도(S)가 이 값 이하(저채도=거의 흰색)에 가까운 부분 선호
            # - min_area  : 너무 작은 노이즈 후보 제거를 위한 최소 면적(픽셀)
            sun_xy = None
            if (f_low_cached is not None) and ok_norm and (f_norm is not None):
                sun_xy = detect_sunlight(
                    f_low_cached, f_norm,
                    v_thr_norm=220,  # 0~255 (HSV V 채널 기준)
                    v_thr_low=200,
                    s_max=40,        # 0~255 (저채도=흰색/하이라이트 성향)
                    min_area=4
                )

            # ----- 5-4) 눈 좌표 추정 (내부 카메라가 있을 때만) -----
            eye_xy = None
            if cam_inside is not None:
                ok_in, f_inside = cam_inside.read()
                if ok_in and f_inside is not None and f_inside.size > 0:
                    eye_xy = detect_eye(f_inside, eye_cascade)
                # 실패 시 eye_xy는 None으로 남음

            # ----- 5-5) 출력 캔버스 생성 & 원 결정/렌더 -----
            # 실제 출력은 '흰색 캔버스'에만 그립니다. 원의 위치는 decide_circle에서 계산.
            out_w, out_h = 1280, 720
            canvas = make_white_canvas(out_w, out_h)

            # decide_circle:
            # - 입력: sun_xy, eye_xy, 광각 프레임의 원본 크기(H, W), 출력 캔버스 크기(out_w, out_h)
            # - 출력: center(tuple|None), intensity(수치)
            #   center   : 최종 원 중심 (없으면 None)
            #   intensity: draw_circle로 넘겨지는 '회색 진하기' 지표.
            #              (내부 구현에 따라 0~1 또는 0~255 등 스케일일 수 있으며,
            #               draw_circle 내부에서 적절히 변환/적용된다고 가정)
            center, intensity = decide_circle(
                sun_xy, eye_xy,
                sun_frame_shape=(H, W, 3),  # 광각 입력 프레임 크기 (행, 열, 채널)
                out_w=out_w,
                out_h=out_h
            )

            # 반지름은 UI/시인성 목적. 프로젝트 요구에 맞게 조정 가능.
            draw_circle(canvas, center, intensity, radius=90)

            # ----- 5-6) 디버그 텍스트(좌측 상단) -----
            overlay_text(canvas, "Sun: {}".format(sun_xy if sun_xy else "None"), y=30,  color=(0, 0, 255))
            overlay_text(canvas, "Eye: {}".format(eye_xy if eye_xy else "None"), y=60,  color=(0, 128, 0))
            overlay_text(canvas, "Expo(L/N) = {}/{}".format(EXPO_LOW, EXPO_NORM), y=90, color=(128, 0, 128))

            # ----- 5-7) 표시/로그 -----
            if use_gui:
                # GUI 모드: 창에 출력
                cv2.imshow("Mapped Circle Output", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Headless 모드: 표준 출력으로 상태 로그
                # intensity는 '원 색의 진하기 지표'이며 draw_circle 내부에서 사용됨.
                print("Sun:", sun_xy, "| Eye:", eye_xy, "| Circle center:", center, "intensity:", intensity)

    finally:
        # ----- 6) 정리 -----
        cam_wide.release()
        if cam_inside is not None:
            cam_inside.release()
        if use_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
