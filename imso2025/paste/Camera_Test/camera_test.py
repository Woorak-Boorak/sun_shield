#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

# ===== 프로그램 설정값 =====
CAM_INDEX = 0                # 사용할 카메라 번호 (0 = 첫 번째 카메라)
FRAME_W, FRAME_H = 1280, 720 # 카메라 해상도 설정

# ===== 카메라 상세 설정 (이 값을 조절하세요!) =====
# 자동 노출 끄기 (0: 수동 모드, 1: 자동 모드)
# 수동으로 노출, 밝기 등을 제어하려면 반드시 0으로 설정해야 합니다.
AUTO_EXPOSURE = 0

# 수동 노출 값 (낮을수록 어두워짐)
# 이 값을 50, 100, 150, 200 등으로 바꿔가며 적절한 밝기를 찾으세요.
MANUAL_EXPOSURE = 150

# 밝기 값 (0 ~ 255 사이, 기본값 128)
# 노출로 조절이 부족할 때 미세 조정용으로 사용하세요.
BRIGHTNESS = 128


def run_camera_test():
    """
    카메라를 열고 설정한 뒤, 실시간 영상을 화면에 출력하는 메인 함수
    """
    # Jetson Nano에서는 GStreamer 파이프라인을 사용하는 것이 안정적입니다.
    pipeline = (
        f'v4l2src device=/dev/video{CAM_INDEX} ! '
        f'video/x-raw, width={FRAME_W}, height={FRAME_H}, framerate=30/1 ! '
        'videoconvert ! '
        'video/x-raw, format=BGR ! appsink'
    )
    
    # GStreamer 파이프라인으로 카메라 열기 시도
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # 카메라가 열리지 않았을 경우
    if not cap.isOpened():
        print(f"[오류] 카메라(장치: /dev/video{CAM_INDEX})를 열 수 없습니다.")
        print("-> 카메라가 물리적으로 잘 연결되었는지 확인해 주세요.")
        print("-> 다른 프로그램이 카메라를 사용하고 있는지 확인해 보세요.")
        return

    # --- 카메라 상세 설정 적용 ---
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    
    # --- 설정 확인용 코드 ---
    # 실제 카메라에 적용된 값을 읽어서 터미널에 출력합니다.
    applied_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    applied_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print("--- 카메라 설정 정보 ---")
    print(f"요청한 노출: {MANUAL_EXPOSURE}, 실제 적용된 노출: {applied_exposure}")
    print(f"요청한 밝기: {BRIGHTNESS}, 실제 적용된 밝기: {applied_brightness}")
    print("-------------------------")
    print("카메라를 성공적으로 열었습니다.")
    print("-> 영상 창을 클릭한 후 'q' 키를 누르면 종료됩니다.")

    # 메인 루프: 계속해서 프레임을 읽어와 화면에 표시
    while True:
        ret, frame = cap.read()
        
        # 프레임을 정상적으로 읽지 못한 경우
        if not ret:
            print("[오류] 카메라에서 프레임을 받아올 수 없습니다.")
            break
        
        # 'Camera Test' 라는 이름의 창에 현재 프레임 표시
        cv2.imshow('Camera Test', frame)
        
        # 'q' 키를 누르면 루프를 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용한 자원 정리
    print("프로그램을 종료합니다.")
    cap.release()
    cv2.destroyAllWindows()

# 이 스크립트가 직접 실행될 때만 main() 함수를 호출
if __name__ == '__main__':
    # SSH 환경에서 실행 시, Jetson Nano에 연결된 모니터에 창을 띄우려면
    # 터미널에서 `export DISPLAY=:0` 명령을 먼저 실행해야 합니다.
    run_camera_test()