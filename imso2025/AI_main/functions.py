#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions.py

[역할(헬퍼 모듈)]
- 카메라 열기/설정 (USB UVC + Jetson Nano 환경 가정)
- 노출 수동 전환 및 안정화 프레임 버퍼링
- OpenCV 3.x / 4.x 모두에서 동작하는 contours 추출 래퍼
- 디버그 텍스트와 흰색 캔버스 유틸
- 눈 검출을 위한 Haar Cascade 로더

[환경 가정]
- Jetson Nano + Ubuntu 18.04 + OpenCV 3.2 (시스템 기본)
- UVC USB 카메라 2대 (/dev/video0, /dev/video1)
- MJPG 포맷이 Nano에서 가장 안정적으로 30FPS를 내는 경우가 많음
"""

import cv2
import numpy as np
import time
import os


# ----------------------------------------------------------------------
# Camera / V4L2 helpers
# ----------------------------------------------------------------------
def open_cam(dev_index: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    """
    지정 인덱스의 UVC 카메라를 열고, 해상도/프레임레이트/MJPG 포맷을 설정한다.

    [입력]
    - dev_index : /dev/video* 인덱스 (예: 0 → /dev/video0)
    - w, h      : 원하는 프레임 크기 (ex. 1280x720)
    - fps       : 원하는 FPS (ex. 30)

    [출력]
    - cv2.VideoCapture 객체 (열기 실패 시에도 객체는 반환되므로 .isOpened()로 확인 필수)

    [주의 / 팁]
    - OpenCV 3.2에서는 CAP_V4L2 플래그를 전달하는 생성자 시그니처가 다르므로 기본 생성자로 처리.
    - set()으로 FOURCC= 'MJPG'를 요청하지만, 최종 적용 여부는 드라이버에 의존한다.
      (일부 카메라는 YUYV만 지원 / 일부는 MJPG에서만 30FPS 가능)
    - 워밍업 루프(최대 1초)는 드라이버 초기 버퍼가 안정화될 시간을 주기 위함.
    """
    cap = cv2.VideoCapture(dev_index)  # OpenCV 3.2: CAP_V4L2 명시 인자 미지원

    # 해상도, FPS, FOURCC 세팅 (적용 실패 가능 → .get()으로 사후 확인 가능)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    # MJPG 강제: Nano + USB UVC 조합에서 대체로 가장 안정적/고FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 간단 워밍업: 드라이버/버퍼 안정화 대기 (최대 1초)
    t0 = time.time()
    ok, frm = False, None
    while time.time() - t0 < 1.0:
        ok, frm = cap.read()
        if ok and frm is not None and frm.size > 0:
            break

    return cap


def set_manual_exposure(cap: cv2.VideoCapture, expo_abs: float) -> None:
    """
    카메라를 '수동 노출' 모드로 전환하고 절대 노출값을 설정한다.

    [입력]
    - cap      : VideoCapture
    - expo_abs : 절대 노출 값 (드라이버 단위. UVC 구현에 따라 스케일 다름)

    [동작 원리]
    - UVC 드라이버별 수동/자동 토글 값이 다를 수 있어, 0.25 / 1 두 경로를 모두 시도.
      (일부 환경: 0.25=수동, 0.75=자동 / 다른 환경: 1=수동, 3=자동 등)
    - 설정 직후에는 이미지가 흔들릴 수 있으므로 grab_after_set()으로 안정화 프레임을 버릴 것.
    """
    # 경로 1) 일부 드라이버: 0.25=수동
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass

    # 경로 2) 다른 드라이버: 1=수동 (혹은 3=자동)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except Exception:
        pass

    # 절대 노출 설정 (실제 스케일/적용 여부는 드라이버에 의존)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(expo_abs))


def grab_after_set(cap: cv2.VideoCapture, settle_frames: int = 2):
    """
    설정 변경 직후 불안정한 프레임을 일정 개수 버리고, 그 다음 유효 프레임 1장을 읽어온다.

    [입력]
    - cap           : VideoCapture
    - settle_frames : 먼저 버릴 프레임 수 (노출/포맷 변경 후 안정화용)

    [출력]
    - (ok, frame)
      ok   : True/False
      frame: 유효 BGR 프레임 (실패 시 None)

    [팁]
    - 일부 카메라/드라이버는 설정 후 2~5프레임 정도가 불안정하다.
    - 내부에서 최대 3회까지 유효 프레임 재시도(작은 sleep 포함).
    """
    ok = False
    frame = None

    # 안정화 프레임 버리기
    for _ in range(settle_frames):
        cap.read()

    # 유효 프레임 잡기(최대 3회 시도)
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.01)

    return ok, frame


# ----------------------------------------------------------------------
# OpenCV 3.x / 4.x 호환 contours
# ----------------------------------------------------------------------
def find_contours_compat(binary_img: np.ndarray):
    """
    OpenCV 3.x와 4.x의 반환 시그니처 차이를 흡수하여 contours만 일관되게 반환.

    [입력]
    - binary_img : 단일 채널 이진 이미지 (0/255)

    [출력]
    - contours : list[np.ndarray] (각 원소는 컨투어 좌표 집합)

    [참고]
    - OpenCV 3.x: findContours → (image, contours, hierarchy)
    - OpenCV 4.x: findContours → (contours, hierarchy)
    """
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:   # OpenCV 3.x
        _, contours, _ = res
    else:               # OpenCV 4.x
        contours, _ = res
    return contours


# ----------------------------------------------------------------------
# Draw / Text helpers
# ----------------------------------------------------------------------
def overlay_text(img: np.ndarray, text: str, y: int = 30,
                 color=(0, 0, 255), scale: float = 0.7, thick: int = 2) -> None:
    """
    이미지 좌측 상단 기준 y 위치에 디버그 텍스트 1줄을 그린다.

    [입력]
    - img   : BGR 이미지
    - text  : 출력 문자열
    - y     : 세로 위치(픽셀). x는 고정(좌측 10px)
    - color : BGR 튜플
    - scale : 폰트 스케일
    - thick : 두께

    [폰트]
    - cv2.FONT_HERSHEY_SIMPLEX (가독성 좋고 보편)
    """
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def make_white_canvas(w: int, h: int) -> np.ndarray:
    """
    주어진 크기의 '흰색 BGR 캔버스'를 생성.

    [입력]
    - w, h : 폭, 높이 (픽셀)

    [출력]
    - (h, w, 3) uint8 배열. 모든 채널이 255(흰색).
    """
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    return canvas


# ----------------------------------------------------------------------
# Haar cascade loader (눈 검출용)
# ----------------------------------------------------------------------
def load_eye_cascade():
    """
    눈 검출용 Haar Cascade를 여러 경로 후보에서 탐색해 로드.

    [반환]
    - cv2.CascadeClassifier 객체(성공)
    - None (실패)

    [경로 후보(우분투 18.04 기준 흔한 위치)]
    - /usr/share/opencv/haarcascades/haarcascade_eye.xml
    - /usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml
    - /usr/share/opencv4/haarcascades/haarcascade_eye.xml
    - ./haarcascade_eye.xml (프로젝트 로컬 배치 가능)

    [주의/팁]
    - Nano/배포판에 따라 OpenCV 데이터 디렉터리가 다를 수 있다.
      위 경로에 없으면 패키지 설치 상태를 확인하거나 xml을 프로젝트 루트에 두자.
    - Cascade 기반 검출은 조명/각도/해상도에 민감. 사전 전처리(equalizeHist 등)를 권장.
    """
    candidates = [
        "/usr/share/opencv/haarcascades/haarcascade_eye.xml",
        "/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_eye.xml",
        "./haarcascade_eye.xml",
    ]
    for path in candidates:
        if os.path.exists(path):
            cas = cv2.CascadeClassifier(path)
            if not cas.empty():
                return cas
    return None
