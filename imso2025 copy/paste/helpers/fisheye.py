#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fisheye.py
----------
광각/어안 카메라 프레임에서 "유효 원형 영상(ROI)"의 중심과 반지름을 추정하고,
해당 원형 ROI에 대한 마스크를 생성하는 유틸 함수 모음.

배경
- 많은 광각/어안 USB 카메라의 출력은 사각 프레임 내부에 '원형'으로 실제 영상이 들어가고,
  원 밖(모서리 영역)은 완전히 검거나 유의미하지 않은 값으로 채워지는 경우가 많습니다.
- 본 모듈은 그 원형 영역을 자동으로 추정(윤곽 기반)하고, 후속 처리에서 ROI만 다루도록
  마스크를 제공하여 성능/정확도를 개선합니다.

구성
- _find_contours_compat: OpenCV 3.x/4.x 버전 차이를 흡수하여 contours/hierarchy를 일관되게 반환.
- estimate_fisheye_circle: 프레임에서 어안 원형 ROI의 (중심, 반지름)을 추정.
- make_circular_mask: 주어진 (중심, 반지름)에 대해 0/1 원형 마스크를 생성.

참고
- Jetson Nano + OpenCV 3.2 환경에서도 정상 동작하도록 호환성을 고려했습니다.
- 경계부 노이즈/비네팅을 피하기 위해 마스크 반지름을 약간 축소(scale)하는 옵션을 제공합니다.
"""

import cv2
import numpy as np
from typing import Tuple


def _find_contours_compat(binary_img):
    """
    OpenCV 버전에 상관없이 contours, hierarchy를 일관된 형태로 얻기 위한 헬퍼.

    OpenCV 버전별 findContours 반환 형태:
    - OpenCV 3.x: (image, contours, hierarchy)
    - OpenCV 4.x: (contours, hierarchy)

    Args:
        binary_img (np.ndarray): 이진 영상(단일 채널). 보통 threshold나 Canny 등의 결과.

    Returns:
        Tuple[list, np.ndarray]: contours(윤곽 리스트), hierarchy(계층 정보)
    """
    # 가장 바깥 윤곽선만 필요하므로 RETR_EXTERNAL 사용,
    # CHAIN_APPROX_SIMPLE로 포인트 수를 줄여 메모리 절약.
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 버전 차이 보정: 길이가 3이면 OpenCV 3.x, 2이면 OpenCV 4.x
    if len(res) == 3:
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res

    return contours, hierarchy


def estimate_fisheye_circle(frame_bgr: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """
    입력 BGR 프레임에서 '어안(원형) 유효 영상'의 중심과 반지름을 추정합니다.

    방법:
    1) 그레이스케일 변환 후, 아주 낮은 임계값(예: 5)로 이진화하여
       완전한 검정(프레임 외곽/비유효 영역)을 제외한 '실제 영상' 영역을 추출합니다.
    2) 추출된 가장 큰 윤곽선을 감싸는 최소 외접원(minEnclosingCircle)을 계산합니다.
       -> 어안 원형 영역을 근사적으로 포착할 수 있습니다.
    3) 안전 장치로 기본 중심/반지름(프레임 중앙, min(w,h)//2)보다 커지지 않도록 제한합니다.

    Args:
        frame_bgr (np.ndarray): 입력 컬러 프레임(BGR, HxWx3)

    Returns:
        Tuple[Tuple[int, int], int]:
            - (cx, cy): 추정된 원 중심 좌표 (정수)
            - r: 추정된 원 반지름 (정수)

    주의:
    - 일부 카메라에서는 원형이 아니라 타원/사각 형태의 마스킹일 수 있습니다.
      이 경우에도 최소 외접원으로 근사합니다.
    - 임계값 5는 "거의 검정"을 제외하기 위한 경험적 값이며, 장면/노출에 따라 조정 가능.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    r_default = min(cx, cy)  # 프레임에 내접하는 기본 최대 원 반지름

    # 1) 저임계 이진화로 '실제 영상' 영역만 남기기
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # 임계값 5: 픽셀 값이 5 초과이면 '유효', 이하는 '무효(검정 배경)'로 간주
    _, th = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # 2) 윤곽선 추출 및 가장 큰 윤곽 선택
    cnts, _ = _find_contours_compat(th)
    if not cnts:
        # 윤곽 자체가 없다면(프레임이 전반적으로 어두운 경우 등) 중앙/기본 반지름 반환
        return (cx, cy), r_default

    c = max(cnts, key=cv2.contourArea)

    # 3) 최소 외접원 -> (중심, 반지름) 계산
    (fx, fy), fr = cv2.minEnclosingCircle(c)
    fx, fy, fr = int(fx), int(fy), int(fr)

    # 너무 큰 값이 나오면 기본값으로 클램프(안전장치)
    fr = min(fr, r_default)

    return (fx, fy), fr


def make_circular_mask(h: int, w: int, center, radius: int, scale: float = 0.98) -> np.ndarray:
    """
    주어진 (중심, 반지름)에 대해 0/1의 원형 마스크를 생성합니다.

    특징:
    - scale 파라미터로 반지름을 약간 축소하여 경계부(검은 테두리/비네팅/왜곡 잔여물)를
      보수적으로 제외할 수 있습니다. 기본 0.98은 98% 크기만 사용한다는 의미.

    구현 메모:
    - np.ogrid를 사용해 (메모리 효율적인) 브로드캐스팅으로 원 방정식(x-cx)^2+(y-cy)^2 <= r^2 평가.
    - 반환은 uint8 배열이며, ROI 내부=1, 외부=0 형태.

    Args:
        h (int): 프레임 높이
        w (int): 프레임 너비
        center (Tuple[int,int]): 원 중심 (cx, cy)
        radius (int): 원 반지름
        scale (float, optional): 반지름 축소 비율(0~1). 기본 0.98.

    Returns:
        np.ndarray: (h, w) 형태의 uint8 마스크. ROI 내부=1, 외부=0
    """
    cx, cy = center
    # 경계부 노이즈를 피하기 위해 반지름을 scale만큼 축소
    r = int(radius * scale)

    # 좌표 그리드 생성(메모리 효율적인 ogrid 사용)
    Y, X = np.ogrid[:h, :w]

    # 원 내부 여부 계산: (x - cx)^2 + (y - cy)^2 <= r^2
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r

    # True/False -> 1/0 (uint8) 변환
    return mask.astype(np.uint8)
