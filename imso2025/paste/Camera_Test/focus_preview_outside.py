#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

def main():
    # 카메라 장치 번호 (보통 /dev/video0 이면 0)
    cap = cv2.VideoCapture(0)

    # 해상도 설정 (1280x720 추천)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    print("[INFO] q 를 눌러 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            break

        # 영상 출력
        cv2.imshow("Wide Camera Preview", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
