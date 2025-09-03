#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os
from datetime import datetime

# ===== 프로그램 설정값 =====
CAM_INDEX = 0            # 사용할 카메라 번호 (0 = 첫 번째 카메라)
FRAME_W, FRAME_H = 1280, 720  # 카메라 해상도 설정
OUTPUT_DIR = "./captures"     # 캡처한 이미지를 저장할 폴더

# 밝기 임계값: 이것보다 어두우면 "밝은 점이 없다"고 판단
MIN_INTENSITY = 0.25     # 0.0(완전 검은색) ~ 1.0(완전 흰색)

def find_contours_compat(binary_img):
    """
    OpenCV 버전이 달라도 똑같이 윤곽선을 찾는 함수
    - OpenCV 3.x와 4.x에서 findContours 함수의 결과가 다르기 때문
    """
    # 이진 이미지에서 윤곽선 찾기
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # OpenCV 버전에 따라 결과 개수가 다름
    if len(res) == 3:     # OpenCV 3.x: (원본이미지, 윤곽선, 계층정보)
        _, contours, hierarchy = res
    else:                 # OpenCV 4.x: (윤곽선, 계층정보)
        contours, hierarchy = res
    return contours, hierarchy

def ensure_dir(path: str) -> None:
    """
    폴더가 없으면 만드는 함수
    """
    if not os.path.exists(path):
        os.makedirs(path)

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    카메라를 열고 설정하는 함수
    """
    # 카메라 연결 시도
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        return None

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 비디오 압축 방식을 MJPG로 설정 (더 빠른 처리를 위해)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 초당 30프레임으로 설정

    # =======================================================
    # ▼▼▼▼▼▼▼▼▼▼ 노출 및 밝기 설정 추가 ▼▼▼▼▼▼▼▼▼▼
    # =======================================================
    # 1. 자동 노출 기능 끄기 (0: 수동, 1: 자동)
    #    이걸 먼저 꺼야 수동 노출 설정이 적용됩니다.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 💡 중요!

    # 2. 수동 노출값 설정 (값이 낮을수록 어두워짐)
    #    카메라마다 지원하는 값의 범위가 다르므로 150, 100, 50 등으로 바꿔보며 테스트하세요.
    cap.set(cv2.CAP_PROP_EXPOSURE, 5000)

    # 3. (선택 사항) 밝기 조절 (0~255 사이 값, 기본값 128)
    #    노출 설정으로 충분하지 않을 때 미세 조정용으로 사용하세요.
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 120)

    # --- 설정 확인용 코드 ---
    # 설정한 값이 실제 카메라에 적용되었는지 확인하기 위해 현재 값을 읽어옵니다.
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print(f"[카메라 설정] 적용된 노출: {exposure}, 밝기: {brightness}")
    # =======================================================
    # ▲▲▲▲▲▲▲▲▲▲ 노출 및 밝기 설정 추가 ▲▲▲▲▲▲▲▲▲▲
    # =======================================================
    
    return cap if cap.isOpened() else None

def estimate_fisheye_circle(frame_bgr: np.ndarray):
    """
    광각/어안 카메라에서 실제 영상이 보이는 원형 영역을 찾는 함수
    - 광각 카메라는 보통 사각형 화면 안에 원형으로 영상이 나타남
    - 이 원의 중심과 반지름을 찾아서 반환
    
    Args:
        frame_bgr: 컬러 이미지
    
    Returns:
        ((중심x, 중심y), 반지름)
    """
    h, w = frame_bgr.shape[:2]  # 이미지 높이, 너비
    cx, cy = w // 2, h // 2     # 화면 중앙 좌표
    r_default = min(cx, cy)     # 기본 반지름 (화면에 맞는 최대 원)

    # 컬러 이미지를 흑백으로 변환
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 완전히 검은 부분(값이 5 이하)을 제거하여 실제 영상 영역만 남김
    _, th = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    # 흰색 영역의 윤곽선을 모두 찾기
    cnts, _ = find_contours_compat(th)

    # 윤곽선이 없으면 화면 중앙에 기본 원 반환
    if not cnts:
        return (cx, cy), r_default

    # 가장 큰 윤곽선 찾기 (보통 이게 어안 렌즈의 원형 영역)
    c = max(cnts, key=cv2.contourArea)
    
    # 이 윤곽선을 감싸는 가장 작은 원 구하기
    (fx, fy), fr = cv2.minEnclosingCircle(c)
    fx, fy, fr = int(fx), int(fy), int(fr)
    
    # 반지름이 너무 크면 기본값으로 제한
    fr = min(fr, r_default)
    return (fx, fy), fr

def make_circular_mask(h: int, w: int, center, radius: int, scale: float = 0.98) -> np.ndarray:
    """
    원형 마스크를 만드는 함수
    - 지정된 원 안쪽은 1, 바깥쪽은 0인 마스크
    - 테두리 노이즈를 피하기 위해 scale로 살짝 줄임
    
    Args:
        h, w: 이미지 높이, 너비
        center: 원의 중심 (x, y)
        radius: 원의 반지름
        scale: 원 크기 조절 (0.98 = 98% 크기)
    
    Returns:
        0과 1로 이루어진 마스크 배열
    """
    cx, cy = center
    r = int(radius * scale)  # 실제 사용할 반지름
    
    # 이미지의 모든 픽셀 좌표 생성
    Y, X = np.ogrid[:h, :w]
    
    # 각 픽셀이 원 안에 있는지 계산 (피타고라스 정리)
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    
    # True/False를 1/0으로 변환
    return mask.astype(np.uint8)

def find_brightest_point(frame_bgr: np.ndarray):
    """
    이미지에서 가장 밝은 점을 찾는 핵심 함수
    
    Args:
        frame_bgr: 분석할 컬러 이미지
    
    Returns:
        - point: 가장 밝은 점의 (x, y) 좌표 (없으면 None)
        - intensity: 밝기 정도 0.0~1.0 (없으면 None)  
        - info: 추가 정보 (어안 렌즈 중심, 반지름 등)
    """
    # 이미지가 비어있으면 분석 불가
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None, {}

    h, w = frame_bgr.shape[:2]
    
    # 1단계: 어안 렌즈의 유효 영역(원) 찾기
    (cx, cy), rad = estimate_fisheye_circle(frame_bgr)
    
    # 2단계: 원형 마스크 만들기 (원 안쪽만 분석하기 위해)
    mask01 = make_circular_mask(h, w, (cx, cy), rad, scale=0.98)

    # 3단계: BGR을 HSV로 변환 (H=색상, S=채도, V=명도)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].copy()  # V채널(명도)만 추출
    
    # 4단계: 원 바깥쪽은 모두 0(검은색)으로 만들기
    V[mask01 == 0] = 0

    # 5단계: 노이즈 줄이기 (7x7 가우시안 블러)
    V_blur = cv2.GaussianBlur(V, (7, 7), 0)

    # 6단계: 가장 밝은 점 찾기
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(V_blur)
    # minVal, maxVal: 최소값, 최대값 (0~255)
    # minLoc, maxLoc: 최소값 위치, 최대값 위치
    
    # 7단계: 밝기를 0.0~1.0 범위로 변환
    intensity = float(maxVal) / 255.0

    # 8단계: 너무 어두우면 "없음"으로 처리
    if intensity < MIN_INTENSITY:
        return None, None, {"center": (cx, cy), "radius": rad}

    # 가장 밝은 점의 좌표와 밝기 반환
    return maxLoc, intensity, {"center": (cx, cy), "radius": rad}

def annotate_result(frame_bgr: np.ndarray, point, intensity: float, info: dict):
    """
    분석 결과를 이미지에 그려서 보여주는 함수
    - 어안 렌즈 영역을 노란 원으로 표시
    - 가장 밝은 점을 빨간 점으로 표시
    - 좌표와 밝기 수치를 텍스트로 표시
    
    Args:
        frame_bgr: 원본 이미지
        point: 밝은 점 좌표
        intensity: 밝기
        info: 추가 정보
    
    Returns:
        표시가 그려진 이미지
    """
    out = frame_bgr.copy()  # 원본을 복사해서 작업
    h, w = out.shape[:2]
    
    # 어안 렌즈 정보 가져오기
    cx, cy = info.get("center", (w // 2, h // 2))  # 중심
    r = info.get("radius", min(w, h) // 2)         # 반지름

    # 어안 렌즈 유효 영역을 노란색 원으로 그리기
    cv2.circle(out, (cx, cy), r, (0, 255, 255), 2, cv2.LINE_AA)
    
    # 원의 중심에 파란색 십자가 표시
    cross_size = 15  # 십자가 크기
    cv2.line(out, (cx - cross_size, cy), (cx + cross_size, cy), (255, 0, 0), 2, cv2.LINE_AA)  # 가로선
    cv2.line(out, (cx, cy - cross_size), (cx, cy + cross_size), (255, 0, 0), 2, cv2.LINE_AA)  # 세로선
    
    # 중심점에 작은 파란색 원점 추가
    cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1, cv2.LINE_AA)

    if point is not None and intensity is not None:
        # 밝은 점을 찾은 경우
        x, y = point
        
        # 빨간색 원점으로 위치 표시
        cv2.circle(out, (x, y), 8, (0, 140, 255), -1, cv2.LINE_AA)
        
        # 좌표와 밝기 정보를 텍스트로 표시
        txt = f"Brightest: ({x},{y})  V={intensity:.3f}"
        cv2.putText(out, txt, (max(10, x + 12), max(25, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2, cv2.LINE_AA)
    else:
        # 밝은 점을 못 찾은 경우 경고 메시지
        cv2.putText(out, "No bright spot detected (below threshold)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return out

def main():
    """
    프로그램의 메인 함수 - 실제 실행되는 부분
    """
    # 1단계: 저장 폴더 확인/생성
    ensure_dir(OUTPUT_DIR)

    # 2단계: 카메라 열기
    cap = open_camera(CAM_INDEX, FRAME_W, FRAME_H)
    if cap is None:
        print(f"[ERROR] Camera #{CAM_INDEX} (/dev/video{CAM_INDEX}) open failed.")
        print("카메라가 연결되어 있는지, 다른 프로그램에서 사용 중이지 않은지 확인해주세요.")
        return

    # 3단계: 사용자에게 사용법 안내
    print("[INFO] Press <Enter> to capture and analyze. Type 'q' then <Enter> to quit.")
    print("사용법: 엔터키를 누르면 사진을 찍어서 분석합니다. 'q' + 엔터키로 종료.")
    
    # 4단계: 메인 루프
    while True:
        # 사용자 입력 받기
        cmd = input("> ").strip().lower()
        if cmd == "q":
            print("[INFO] Quit requested.")
            break

        # 카메라에서 한 프레임 읽기
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame. Try again.")
            print("카메라에서 이미지를 읽지 못했습니다. 다시 시도해주세요.")
            continue

        # 현재 시간으로 파일명 생성
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = os.path.join(OUTPUT_DIR, f"capture_{ts}.jpg")
        
        # 원본 이미지 저장
        cv2.imwrite(raw_path, frame)

        # 가장 밝은 점 분석
        point, intensity, info = find_brightest_point(frame)
        
        # 결과 출력
        if point is not None:
            print(f"[RESULT] Brightest at {point}, intensity={intensity:.3f}")
            print(f"가장 밝은 점: 좌표 {point}, 밝기 {intensity:.3f}")
        else:
            print("[RESULT] No bright spot (below threshold).")
            print(f"밝은 점을 찾지 못했습니다. (임계값 {MIN_INTENSITY} 이하)")

        # 분석 결과가 그려진 이미지 생성
        vis = annotate_result(frame, point, intensity, info)
        ann_path = os.path.join(OUTPUT_DIR, f"capture_{ts}_annotated.jpg")
        
        # 분석 결과 이미지 저장
        cv2.imwrite(ann_path, vis)

        print(f"[INFO] Saved: {raw_path}")
        print(f"[INFO] Saved: {ann_path}")
        print("원본 이미지와 분석 결과 이미지를 저장했습니다.\n")

    # 5단계: 정리 작업
    cap.release()  # 카메라 해제
    print("프로그램을 종료합니다.")

# 프로그램 시작점
if __name__ == "__main__":
    main()