#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
maxima.py
---------
밝기(또는 단일 채널 스코어) 이미지에서 상위 K개 '최댓값 지점'을 추출하고,
결과를 시각적으로 주석(ROI 원, 중심, Top1~Top3 포인트/라벨)으로 그려주는 유틸.

구성
- top_k_maxima(img, k=3, suppress_radius=15)
  : Non-Maximum Suppression(NMS) 방식으로 상위 k개 최대값 좌표/값을 반환.
- draw_annotations(img, center, radius, peaks)
  : ROI 원, ROI 중심(십자/점), Top1~3 포인트와 해당 V값(0~1 정규화)을 주석으로 그려줌.

설계 메모
- 입력 img는 단일 채널(예: HSV의 V채널). 0~255 범위 가정(OpenCV 표준).
- NMS는 '최댓값 주변 반경을 0으로 억제'하는 매우 단순/빠른 방식으로 구현.
- OpenCV 3.2/4.x 호환. (cv2.minMaxLoc 사용)
"""

import cv2
import numpy as np
from typing import List, Tuple

def find_all_maxima(img: np.ndarray) -> Tuple[List[Tuple[int,int]], int]:
    """
    단일채널 이미지에서 전역 최대값(maxVal)과 동일한 모든 픽셀 좌표를 반환.
    반환: (points, maxVal)
      - points: [(x,y), ...]  # 최대값인 모든 픽셀들
      - maxVal: 0~255 정수 최대값
    """
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
    maxVal = int(maxVal)
    if maxVal <= 0:
        return [], 0

    ys, xs = np.where(img == maxVal)      # (y,x) 순
    points = [(int(x), int(y)) for y, x in zip(ys, xs)]  # (x,y) 변환
    return points, maxVal

def top_k_maxima(img: np.ndarray, k: int = 3, suppress_radius: int = 15) -> List[Tuple[Tuple[int,int], int]]:
    """
    블러링/스무딩이 적용된 단일 채널 이미지에서 상위 k개의 '최댓값' 위치와 값을 찾는다.
    간단한 Non-Maximum Suppression(NMS)으로 각 포인트가 서로 겹치지 않게 보장한다.

    동작 개요:
    1) 작업용 복사본(work)을 만든다.
    2) minMaxLoc으로 현재 최대값 위치/값을 얻는다.
    3) 그 최대값을 결과 리스트에 추가하고, 해당 점 주변 suppress_radius 이내를 0으로 만든다.
    4) 위 과정을 k번 반복(또는 더 이상 최대값이 없을 때까지)한다.

    Args:
        img (np.ndarray): 단일 채널(0~255 범위)의 입력 이미지 (예: HSV V채널).
        k (int): 추출할 최대값 포인트 개수 (기본 3).
        suppress_radius (int): 비최대 억제 반경(픽셀). 값이 클수록 포인트 간 간격이 넓어짐.

    Returns:
        List[Tuple[Tuple[int,int], int]]:
            [ ((x,y), value_0_255), ... ] 형태의 리스트.
            - 좌표는 (x, y), 값은 0~255 정수.
            - 최대 k개, 이미지에 충분한 피크가 없으면 더 적을 수 있음.

    주의:
    - 입력이 float형이더라도 minMaxLoc은 동작하지만, 여기서는 0~255 정수형을 가정하고
      텍스트 표시에 사용하기 위해 int로 캐스팅한다.
    - 억제 반경은 경험적으로 조정. 너무 작으면 중복 포인트가 인접해서 뽑힐 수 있다.
    """
    # 원본을 보존하기 위해 복사본에서 작업
    work = img.copy()
    h, w = work.shape[:2]
    peaks: List[Tuple[Tuple[int,int], int]] = []

    # 브로드캐스팅을 위한 좌표 격자(메모리 효율적인 ogrid 사용)
    yy, xx = np.ogrid[:h, :w]

    for _ in range(k):
        # 현재 남은 픽셀 중 최댓값/좌표
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(work)
        if maxVal <= 0:
            # 남은 유효 최대값이 없으면 종료
            break

        x, y = maxLoc
        peaks.append(((x, y), int(maxVal)))

        # 원형 반경 suppress_radius 이내를 0으로 만들어 다음 반복에서 제외(NMS)
        # (x - x0)^2 + (y - y0)^2 <= r^2 내에 True인 마스크 생성
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= suppress_radius * suppress_radius
        work[mask] = 0  # 해당 영역을 0으로 억제

    return peaks


# 파일: helpers/maxima.py

def draw_annotations_allmax(
    img: np.ndarray,
    center: Tuple[int,int],
    radius: int,
    points: List[Tuple[int,int]],
    max_val_255: int
) -> np.ndarray:
    """
    ROI 원/중심을 그리고, 전역 최대값과 동일한 모든 좌표를 같은 색으로 표시.
    좌상단에는 intensity와 count를 표기.
    """
    out = img.copy()

    # ROI 원(노란색)
    cv2.circle(out, center, radius, (0, 255, 255), 2, cv2.LINE_AA)

    # 중심 십자/점(파란색)
    cx, cy = center
    cross = 15
    cv2.line(out, (cx - cross, cy), (cx + cross, cy), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(out, (cx, cy - cross), (cx, cy + cross), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1, cv2.LINE_AA)

    # 최대값 픽셀 모두 표시(주황 계열)
    for (x, y) in points:
        cv2.circle(out, (x, y), 8, (0, 140, 255), -1, cv2.LINE_AA)

    # 요약 텍스트
    if max_val_255 > 0:
        v = max_val_255 / 255.0
        cv2.putText(
            out, f"Top intensity={v:.3f} (count={len(points)})",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 200), 2, cv2.LINE_AA
        )

    return out
