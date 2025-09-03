#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSH로 접속 중이어도, 보드에 직결된 모니터(:0)에 전체화면으로
흰 배경을 띄우고, 터미널에 입력하는 좌표(x y [r])로 작은 원을 표시합니다.

사용법(예):
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
python3 OpenCV_GUI.py --width 1280 --height 720 --fullscreen 1 --radius 20

터미널에 아래처럼 입력하면 즉시 반영됩니다:
100 200
640 360 30
q  (종료)
"""

import argparse
import sys
import time
import cv2
import numpy as np
import select

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_line(line, W, H, default_r):
    """
    입력 형태:
      - 'x y'           -> 기본 반지름(default_r) 사용
      - 'x y r'         -> 반지름 지정
    잘못된 입력은 None 반환
    """
    try:
        xs = line.strip().split()
        if len(xs) == 0:
            return None
        # 종료 키워드
        if xs[0].lower() in ["q", "quit", "exit"]:
            return "QUIT"

        if len(xs) < 2:
            return None

        x = int(float(xs[0]))
        y = int(float(xs[1]))
        r = default_r if len(xs) < 3 else int(float(xs[2]))

        return x, y, r
    except:
        return None

def main():
    ap = argparse.ArgumentParser(description="OpenCV 전체화면 + 좌표 입력으로 원 표시")
    ap.add_argument("--width",  type=int, default=1280, help="출력 너비")
    ap.add_argument("--height", type=int, default=720,  help="출력 높이")
    ap.add_argument("--window", default="Circle",       help="윈도우 이름")
    ap.add_argument("--fullscreen", type=int, default=1, help="1=전체화면, 0=창모드")
    ap.add_argument("--radius", type=int, default=20,   help="좌표만 입력했을 때 기본 반지름")
    args = ap.parse_args()

    W, H = args.width, args.height
    default_r = max(1, args.radius)

    print("[INFO] OpenCV:", cv2.__version__)
    print("[INFO] 해상도: {}x{}".format(W, H))
    print("[INFO] 'x y [r]' 형태로 좌표를 입력하세요. 'q' 입력 시 종료합니다.")
    print("[INFO] 예) 640 360    또는    640 360 30")

    # 초기 상태: 화면 중앙에 기본 반지름
    cx, cy, cr = W // 2, H // 2, default_r

    # 윈도우 생성/전체화면 설정
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    if args.fullscreen == 1:
        cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 렌더 루프
    while True:
        # 1) 흰 배경
        img = np.full((H, W, 3), 255, dtype=np.uint8)

        # 2) 원 그리기(검정)
        #   - 화면 밖으로 나가지 않도록 중심과 반지름을 안전 범위로 보정
        x = clamp(cx, 0, W)
        y = clamp(cy, 0, H)
        r = max(1, cr)
        cv2.circle(img, (x, y), r, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        # 3) 표시
        cv2.imshow(args.window, img)

        # 4) 키보드 'q' 종료 (모니터에 키보드가 연결되어 있을 때)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        # 5) 터미널(표준입력)에서 좌표 읽기 (non-blocking)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if rlist:
            line = sys.stdin.readline()
            if not line:  # EOF
                pass
            else:
                parsed = parse_line(line, W, H, default_r)
                if parsed == "QUIT":
                    break
                if isinstance(parsed, tuple):
                    nx, ny, nr = parsed
                    # 화면 범위로 클램핑
                    cx = clamp(nx, 0, W)
                    cy = clamp(ny, 0, H)
                    cr = max(1, nr)

        # 6) 루프가 너무 빨라지지 않게 살짝 쉼
        time.sleep(0.002)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
