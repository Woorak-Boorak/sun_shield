#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core.py

[역할]
- (1) 광각 카메라의 두 노출 프레임(저/보통)을 이용해 '진짜 강한 광원(햇빛)' 좌표를 추정
- (2) 내부 카메라 프레임에서 Haar Cascade로 '눈(eye)' 좌표를 추정
- (3) 추정된 좌표들을 바탕으로 출력 캔버스에 표시할 '원(circle)의 위치와 명암(intensity)'을 결정
- (4) 실제로 캔버스에 원을 그림

[용어]
- frame_low  : 저노출(밝은 영역이 덜 포화되는) 프레임
- frame_norm : 보통 노출(시청/표시용) 프레임
- sun_xy     : 햇빛(광원) 좌표 (x, y). 없으면 None
- eye_xy     : 눈 좌표 (x, y). 없으면 None
- intensity  : 원의 '회색 밝기값(0~255)' — 값이 클수록 밝은 회색(흰색에 가까움)

[주의]
- 현재 decide_circle()에서는 eye_xy(눈 좌표)를 '미사용' 상태로 두고 있음.
  원의 위치는 sun_xy(햇빛 좌표)만으로 매핑하고, 명암은 '화면 중심과의 거리'로만 계산한다.
  프로젝트 요구에 맞추어, 추후 eye_xy 기반 가중/보정을 추가할 수 있음(TODO 참고).
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from functions import find_contours_compat  # OpenCV 3.x/4.x 호환 contour 추출 래퍼
#  - 가정: 반환값은 contours(list of np.ndarray). hierarchy는 내부에서 숨김


# =========================
# 1) 광각 카메라 → 햇빛(광원) 위치 반환
# =========================
def detect_sunlight(
    frame_low,
    frame_norm,
    v_thr_norm: int = 220,
    v_thr_low: int = 200,
    s_max: int = 40,
    min_area: int = 4
) -> Optional[Tuple[int, int]]:
    """
    두 노출(낮음/보통) 프레임을 이용하여 '두 프레임 모두 매우 밝고 채도가 낮은' 영역의 교집합(core)을 찾고,
    그 후보들 중 '국소 대비(local contrast)'와 '원형성(roundness)'이 높은 컨투어의 중심을 반환한다.

    [입력]
    - frame_low  : 저노출 BGR 프레임 (햇빛처럼 매우 밝은 영역이 여전히 밝게 남아 있는지 확인용)
    - frame_norm : 보통 노출 BGR 프레임 (기본 후보 추출 및 채도(s) 판단용)
    - v_thr_norm : (0~255) 보통 노출 프레임에서의 밝기(V) 임계값
    - v_thr_low  : (0~255) 저노출 프레임에서의 밝기(V) 임계값
                   → 두 프레임 모두 임계 이상일 때 '진짜 강한 하이라이트'로 신뢰 ↑
    - s_max      : (0~255) 채도(S) 상한. 너무 유채색인(채도가 높은) 영역은 제외하고
                   '거의 흰색(하이라이트 성향)'인 영역만 선호
    - min_area   : 컨투어 최소 면적(px). 아주 작은 노이즈 제거용

    [출력]
    - (cx, cy)  : 최고 스코어 후보의 중심 좌표 (정수)
    - None      : 적합한 후보가 없을 때

    [스코어 구성]
    - local_contrast (0~큰 값): 중심 주변 패치의 '최대 밝기' / '주변(고리) 상위 95% 밝기' 비율
      → 중심이 주변보다 얼마나 두드러지게 밝은가
    - roundness (0~1): 컨투어에 속한 점들의 공분산 행렬 고유값 비율 λ_min / (λ_max+ε)
      → 1에 가까울수록 둥근 분포(원형)에 가까움
    - 최종 score = 0.7*local_contrast + 0.3*roundness
    """
    if frame_low is None or frame_norm is None:
        return None

    # ---- 1) 그레이 변환 + 가우시안 블러(작은 노이즈 완화) ----
    blur = 3  # 3x3 권장 (5x5 이상은 과도하게 번질 수 있음)
    g_low  = cv2.GaussianBlur(cv2.cvtColor(frame_low,  cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    g_norm = cv2.GaussianBlur(cv2.cvtColor(frame_norm, cv2.COLOR_BGR2GRAY), (blur, blur), 0)

    # ---- 2) 두 프레임에서 각각 '아주 밝은 영역' 이진 마스크화 ----
    #  - b1: 보통 노출에서 매우 밝은 픽셀
    #  - b2: 저노출에서도 여전히 밝은 픽셀
    _, b1 = cv2.threshold(g_norm, v_thr_norm, 255, cv2.THRESH_BINARY)
    _, b2 = cv2.threshold(g_low,  v_thr_low,  255, cv2.THRESH_BINARY)

    # ---- 3) 교집합(core) + 작은 잡음 제거(개열기: MORPH_OPEN) ----
    core = cv2.bitwise_and(b1, b2)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ---- 4) 채도(S) 마스크: 거의 흰색(저채도) 영역만 남기기 ----
    hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    sat_mask = cv2.inRange(s, 0, s_max)  # S ∈ [0, s_max]
    core = cv2.bitwise_and(core, sat_mask)

    # ---- 5) 컨투어 추출 ----
    contours = find_contours_compat(core)  # [가정] contours(list)만 반환
    if not contours:
        return None

    # ---- 6) 후보별 스코어 계산: local_contrast + roundness ----
    best_score, best_xy = None, None
    h, w = g_norm.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # 너무 작은 노이즈 후보 제거

        # 중심(무게중심) 계산
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # ---- (a) 국소 대비(local contrast) ----
        #  - 중심 주변 '패치'에서의 최대 밝기(local_max)
        #  - 패치의 바깥 고리(ring)에서의 상위 95% 밝기(ring_p95)
        #    → 중심이 주변보다 어느 정도 두드러지는지 비율로 평가
        r = 15  # 패치 반경 (px)
        x0 = max(0, cx - r); x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r); y1 = min(h, cy + r + 1)
        patch = g_norm[y0:y1, x0:x1]
        local_max = float(np.max(patch))

        yy, xx = np.ogrid[y0:y1, x0:x1]
        rr = (yy - cy) ** 2 + (xx - cx) ** 2
        # 고리 두께: [6, 15] px 범위. 충분한 표본이 없으면 패치 자체로 대체
        ring = patch[(rr >= 6**2) & (rr <= 15**2)]
        ring_p95 = np.percentile(ring, 95) if ring.size >= 10 else np.percentile(patch, 95)
        local_contrast = (local_max + 1.0) / (ring_p95 + 1.0)  # 1.0 오프로 0-분모 보호

        # ---- (b) 원형성(roundness) ----
        #  - 컨투어 내부 픽셀 좌표의 공분산 행렬 고유값(λ_min, λ_max) 비율
        #  - 둥글수록 두 축 분산이 비슷 → λ_min/λ_max ≈ 1
        mask = np.zeros_like(core)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # 컨투어 내부 채우기
        ys, xs = np.where(mask > 0)
        if len(xs) >= 5:
            pts = np.vstack([xs, ys]).astype(np.float32).T  # (N, 2)
            cov = np.cov(pts, rowvar=False)
            evals, _ = np.linalg.eig(cov)
            evals = np.sort(evals)  # [λ_min, λ_max]
            roundness = float(evals[0] / (evals[1] + 1e-6))  # 0~1
        else:
            # 표본 너무 적을 때는 보수적 기본값
            roundness = 0.3

        # ---- (c) 최종 스코어 & 최고 후보 갱신 ----
        score = 0.7 * local_contrast + 0.3 * roundness
        if (best_score is None) or (score > best_score):
            best_score, best_xy = score, (cx, cy)

    return best_xy


# =========================
# 2) 내부 카메라 → 눈 위치 반환
# =========================
def detect_eye(frame_color, eye_cascade) -> Optional[Tuple[int, int]]:
    """
    Haar Cascade를 사용하여 가장 '큰 눈 후보' 1개를 찾고, 그 중심 좌표를 반환한다.

    [입력]
    - frame_color : BGR 프레임
    - eye_cascade : cv2.CascadeClassifier (haarcascade_eye.xml 로드 결과)

    [출력]
    - (cx, cy) : 가장 큰 눈 후보의 중심 좌표
    - None     : 검출 실패

    [팁]
    - cascade 기반 검출은 조명/각도에 민감하다.
      필요 시 histogram equalization(gray = equalizeHist(gray))로 대비를 평탄화하여 검출률 향상.
    """
    if frame_color is None or eye_cascade is None:
        return None

    # 그레이 + 히스토그램 평활화(검출 안정성↑)
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # detectMultiScale 파라미터:
    # - scaleFactor : 이미지 피라미드 스케일 간격 (1.05~1.2 권장)
    # - minNeighbors: 후보 합치기 임계 (클수록 오검출↓, 하지만 검출률도↓)
    # - minSize     : 너무 작은 오검출 제거용 최소 크기
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(20, 20)
    )
    if len(eyes) == 0:
        return None

    # 가장 큰 바운딩 박스를 선택 (면적 기준)
    areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in eyes]
    _, best = max(areas, key=lambda t: t[0])
    x, y, w, h = best
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)


# =========================
# 3) 원의 위치·명암 결정
# =========================
def decide_circle(
    sun_xy: Optional[Tuple[int, int]],
    eye_xy: Optional[Tuple[int, int]],
    sun_frame_shape: Tuple[int, int, int],
    out_w: int,
    out_h: int
) -> Tuple[Tuple[int, int], int]:
    """
    출력 캔버스에 그릴 원의 '중심 좌표'와 '명암(intensity)'을 결정한다.

    [현재 정책]
    - 원 위치: 햇빛 좌표(sun_xy)를 광각 원본 좌표계 → 출력 캔버스 좌표계로 선형 매핑
               (sun_xy가 없으면 화면 중앙)
    - 원 명암: '화면 중심과의 정규화 거리(dist)'가 가까울수록 밝게(값↑), 멀수록 어둡게(값↓) 설정
               intensity = clip( 255 - dist * 255 * 1.2 , 50, 220 )
      ※ 여기서 intensity는 '회색 밝기값(0~255)'이며, draw_circle에서 (i,i,i)로 사용됨.
        즉, 값이 클수록 밝은 회색(흰색에 가까움), 값이 작을수록 어두운 회색(검정에 가까움).

    [향후 확장 아이디어 / TODO]
    - 햇빛의 '강도' 또는 '후보 스코어'를 인자로 받아 intensity에 반영
    - eye_xy와 sun_xy의 상대 위치/각도를 사용해 원의 위치/크기/명암을 동적으로 조절
      (예: 눈이 햇빛 방향과 가까우면 더 어둡게, 멀면 더 밝게 등)
    """
    # ---- 1) 원 중심 좌표 결정 ----
    if sun_xy is None:
        # 햇빛 미검출 시: 화면 중앙에 표시(안내/기본값)
        cx, cy = out_w // 2, out_h // 2
    else:
        # 광각 원본 좌표계 → 출력 좌표계로 정규화 매핑
        sh, sw = sun_frame_shape[:2]  # (height, width)
        nx = np.clip(sun_xy[0] / max(1, sw), 0.0, 1.0)  # 0~1
        ny = np.clip(sun_xy[1] / max(1, sh), 0.0, 1.0)  # 0~1
        cx = int(nx * out_w)
        cy = int(ny * out_h)

    # ---- 2) 원 명암(intensity) 계산 ----
    # 화면 중앙에서의 상대 거리(dist) 기반 단순 감쇠 모델
    # - dist = 0이면 중심(가장 밝게), 커질수록 어둡게
    # - 1.2 배율은 체감 곡선을 조정하는 상수(실험값). 필요 시 조정.
    dx = (cx - out_w / 2) / out_w
    dy = (cy - out_h / 2) / out_h
    dist = np.sqrt(dx * dx + dy * dy)  # 0(중심) ~ 대략 0.7(모서리 근처)

    # intensity(밝기값) 범위를 [50, 220]로 제한:
    # - 0(완전 검정)이나 255(완전 흰색)는 눈부심/식별성 측면에서 불편할 수 있어 제한
    intensity = int(np.clip(255 - dist * 255 * 1.2, 50, 220))

    # NOTE: 현재 eye_xy는 미사용(향후 정책 반영 지점)
    return (cx, cy), intensity


# =========================
# 4) 실제 원 그리기
# =========================
def draw_circle(canvas, center_xy: Tuple[int, int], intensity: int, radius: int = 80):
    """
    흰색 배경(canvas)에 '회색 원'을 채워서 그림.

    [입력]
    - canvas     : BGR 이미지 (배경은 흰색이라고 가정)
    - center_xy  : 원 중심 (cx, cy)
    - intensity  : 회색 밝기값(0~255). (i, i, i)로 사용 → i가 클수록 밝은 회색
    - radius     : 원 반지름(px)

    [출력]
    - canvas 자체를 in-place로 수정 후 반환(관례적으로 return도 함께 제공)
    """
    cx, cy = center_xy
    color = (intensity, intensity, intensity)  # B=G=R 같은 값 → 회색
    cv2.circle(canvas, (cx, cy), radius, color, -1)  # thickness=-1: 채우기
    return canvas
