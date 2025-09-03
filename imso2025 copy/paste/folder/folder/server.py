#!/usr/bin/env python3
# pip install websockets

import asyncio
import re
import os
import subprocess
from pathlib import Path
import websockets
import concurrent.futures

clients = set()
last_y, last_x = 50, 50  # 'hide' 같은 명령에서 사용할 기본 좌표

# === 여기부터 추가: index.html 자동 실행 ===
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = (BASE_DIR / "index.html").resolve()

def open_index_on_display():
    import os, subprocess
    from pathlib import Path
    INDEX_PATH = Path(__file__).with_name("index.html").resolve()
    if not INDEX_PATH.exists():
        print(f"[경고] index.html이 없습니다: {INDEX_PATH}")
        return
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    url = f"file://{INDEX_PATH}"

    candidates = [
        ["chromium-browser", "--disable-gpu", "--no-sandbox", "--use-gl=swiftshader", "--app=" + url],
        ["google-chrome",   "--disable-gpu", "--no-sandbox", "--use-gl=swiftshader", "--app=" + url],
        ["firefox", url],           # 크롬류가 없으면 파폭 시도
        ["xdg-open", url],          # 최후의 보루(환경 따라 GPU옵션 적용 안 됨)
    ]
    for cmd in candidates:
        try:
            subprocess.Popen(cmd, env=env)
            print("브라우저 실행:", " ".join(cmd), f"(DISPLAY={env.get('DISPLAY')})")
            return
        except FileNotFoundError:
            continue
    print(f"[안내] 자동 실행 실패. 수동으로 열어주세요: {url}")
# === 추가 끝 ===

def make_packet(v, y, x) -> str:
    v = 0 if int(v) == 0 else 1
    y = max(0, min(100, int(y)))
    x = max(0, min(100, int(x)))
    return f"({v},{y},{x})"

async def send_to_all(msg: str):
    # 끊긴 클라이언트 정리하면서 전송
    dead = []
    for ws in list(clients):
        try:
            await ws.send(msg)
        except websockets.exceptions.ConnectionClosed:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
async def handler(ws, path):
    clients.add(ws)
    try:
        async for msg in ws:
            print("from client:", msg)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(ws)

def parse_line(line: str):
    """
    지원 입력:
      1) (v,y,x)               예) (1,50,25)
      2) v y x  /  v,y,x       예) 1 50 25   또는 1,50,25
      3) show y x              예) show 40 60  -> (1,40,60)
      4) hide                  예) hide       -> (0,last_y,last_x)
    """
    global last_y, last_x
    s = line.strip().lower()

    if s in ("q", "quit", "exit"):
        raise SystemExit

    if s == "help":
        raise ValueError("형식: (v,y,x) | v y x | v,y,x | show y x | hide | q")

    m = re.fullmatch(r"\((\d+),(\d+),(\d+)\)", s)
    if m:
        v, y, x = map(int, m.groups())
        last_y, last_x = y, x
        return v, y, x

    m = re.fullmatch(r"(\d+)[,\s]+(\d+)[,\s]+(\d+)", s)
    if m:
        v, y, x = map(int, m.groups())
        last_y, last_x = y, x
        return v, y, x

    m = re.fullmatch(r"show[,\s]+(\d+)[,\s]+(\d+)", s)
    if m:
        y, x = map(int, m.groups())
        last_y, last_x = y, x
        return 1, y, x

    if s == "hide":
        return 0, last_y, last_x

    raise ValueError("알 수 없는 입력. help 를 입력해 형식을 확인하세요.")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

async def input_loop():
    print(
        "입력 예) (1,50,25) | 1 50 25 | 1,50,25 | show 40 60 | hide | q\n"
        "※ v=0이면 숨김, v=1이면 표시, y/x는 0~100 (퍼센트)\n"
    )
    loop = asyncio.get_event_loop()
    while True:
        try:
            # asyncio.to_thread → run_in_executor 대체
            line = await loop.run_in_executor(executor, input, "좌표 입력> ")
            if not line:
                continue
            v, y, x = parse_line(line)
            pkt = make_packet(v, y, x)
            await send_to_all(pkt)
            print(f"sent: {pkt}  (clients={len(clients)})")
        except SystemExit:
            print("종료합니다.")
            try:
                await send_to_all(make_packet(0, last_y, last_x))
            finally:
                break
        except ValueError as e:
            print("입력 오류:", e)
        except Exception as e:
            print("예상치 못한 오류:", e)

async def main():
    async with websockets.serve(handler, "localhost", 8000):
        print("WebSocket server on ws://localhost:8000")
        # 브라우저 자동 오픈 (Jetson 모니터 화면)
        open_index_on_display()
        await input_loop()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
