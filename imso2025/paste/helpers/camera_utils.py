#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
camera_utils.py
---------------
Jetson Nano + Ubuntu 18.04 환경에서 USB 카메라 상태 점검과 오픈(초기화)을 담당하는 유틸 모듈.

이 모듈은 다음을 제공합니다.
- probe_camera: 특정 인덱스의 카메라가 실제로 프레임을 제공하는지 빠르게 점검
- check_cameras_or_prompt: 프로젝트에서 사용하는 두 개 카메라(WIDE/INNER)의 인식 현황을 출력
- open_camera: 지정 해상도/포맷(MJPG, 30 FPS 권장)으로 VideoCapture를 열어 반환

참고:
- Jetson Nano + UVC(일반 USB 웹캠) 조합에서 고해상도 실시간 처리는 MJPG를 권장합니다.
- 일부 카메라/드라이버는 설정한 FOURCC/FPS가 무시될 수 있습니다(드라이버/펌웨어/USB대역폭 영향).
- 본 모듈은 GUI 없이 headless로도 동작하도록 설계되었습니다.
"""

import cv2
# 상대 경로 import: helpers 패키지로 인식되려면 상위 폴더에 __init__.py 가 존재해야 함
from .io_utils import FRAME_W, FRAME_H, WIDE_CAM_INDEX, INNER_CAM_INDEX


def probe_camera(index: int, width: int, height: int) -> bool:
    """
    지정한 카메라 인덱스가 '실제로 사용 가능한지' 빠르게 점검합니다.

    동작 방식:
    1) VideoCapture(index)로 장치를 연 후,
    2) 원하는 해상도(width, height)를 설정하고,
    3) cap.read()로 프레임을 1장 읽어봅니다.
       - 프레임을 실제로 읽어야 드라이버/장치가 정상 응답하는지 확실하게 확인 가능.
       - isOpened()만 True이고 read()가 실패하는 경우가 드물게 있음(드라이버/권한/다른 프로세스 점유 등).

    Args:
        index (int): /dev/video{index} (예: 0 -> /dev/video0)
        width (int): 요청할 프레임 가로 해상도
        height (int): 요청할 프레임 세로 해상도

    Returns:
        bool: 정상적으로 프레임을 읽어오면 True, 아니면 False
    """
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        # 장치 자체가 열리지 않는 경우: 케이블 분리, 권한/드라이버 문제, 이미 다른 프로세스가 점유 등
        return False

    # 해상도 지정(장치가 반드시 따를 보장은 없음)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 프레임 1장 읽어서 실제 동작 여부 확인
    ok, _ = cap.read()

    # 자원 반환(점검용이므로 즉시 닫음)
    cap.release()

    return bool(ok)


def check_cameras_or_prompt() -> bool:
    """
    프로젝트에서 사용하는 두 카메라(WIDE, INNER)의 인식 상태를 콘솔에 출력하고,
    하나 이상 미인식이어도 계속 진행합니다(사용자 입력으로 중단시키지 않음).

    설계 배경:
    - 개발/디버그 단계에서 내부 카메라가 연결되지 않은 상태가 빈번할 수 있음.
    - 자동화/헤드리스 환경에서도 멈추지 않고 가능한 범위 내에서 계속 진행하도록 설계.

    출력:
        [INFO] Camera #n (/dev/video{n}): OK | NOT DETECTED
        [WARN] Missing: /dev/videoX, ...  (필요 시)

    Returns:
        bool: 항상 True (현재 정책상 진행을 중단하지 않음)
              * 추후 정책 변경 시, 필수 카메라가 없으면 False를 반환하도록 수정 가능.
    """
    status = {}

    # 두 카메라(광각/내부)를 순회하며 점검
    for idx in (WIDE_CAM_INDEX, INNER_CAM_INDEX):
        detected = probe_camera(idx, FRAME_W, FRAME_H)
        status[idx] = detected
        print(f"[INFO] Camera #{idx} (/dev/video{idx}): {'OK' if detected else 'NOT DETECTED'}")

    if all(status.values()):
        # 두 장치 모두 사용 가능
        print("[INFO] All cameras detected. Proceeding.")
        return True

    # 하나 이상 미인식인 경우: 어떤 장치가 없는지 목록 출력
    missing = [f"/dev/video{idx}" for idx, ok in status.items() if not ok]
    print(f"[WARN] Missing: {', '.join(missing)}. Continuing with available cameras.")

    # 현재 정책: 계속 진행
    return True


def open_camera(index: int, width: int, height: int):
    """
    지정한 카메라를 열고, 권장 설정(MJPG, 30 FPS, 1280x720)을 적용한 뒤 VideoCapture 핸들을 반환합니다.

    주의/제약:
    - 모든 설정은 '요청'일 뿐, 실제로 반영되는지는 장치/드라이버/대역폭에 따라 달라질 수 있습니다.
    - 반환값은 cv2.VideoCapture 객체 또는 None 입니다.
      -> None이면 호출 측에서 적절히 우회/에러 처리해야 합니다.
    - MJPG 설정은 Jetson Nano + UVC 카메라에서 CPU 부하와 USB 대역폭을 고려할 때 실무적으로 유리합니다.
      (YUYV는 무압축이라 화질은 좋지만, 고해상도에서 FPS가 크게 떨어지며 부하가 큼)

    Args:
        index (int): /dev/video{index} (예: 0)
        width (int): 요청 가로 해상도 (예: 1280)
        height (int): 요청 세로 해상도 (예: 720)

    Returns:
        cv2.VideoCapture | None: 성공 시 열린 캡처 객체, 실패 시 None
    """
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        # 다른 프로세스가 이미 카메라를 점유하거나, 권한/드라이버 문제가 있을 수 있음
        return None

    # 해상도 설정 (장치에서 거부/조정될 수 있음)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # FOURCC: Motion-JPEG (권장)
    # - 일부 드라이버는 이 설정을 무시할 수 있음
    # - MJPG가 적용되면 동일 해상도/프레임레이트에서 CPU/USB 사용량이 상대적으로 안정적
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # FPS 요청(실제 적용은 장치 영향)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # (선택) 추가 튜닝 예시: 드롭/지연 최소화
    # -> OpenCV 3.2에서는 일부 속성이 무시될 수 있음. 필요 시 테스트 후 사용하세요.
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # 버퍼를 작게(프레임 지연 감소)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 수동/우선 제어 (드라이버마다 상이)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6)        # 노출값(카메라마다 단위/범위 다름)

    return cap
