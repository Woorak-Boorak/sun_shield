# server_manual.py
# pip install websockets

import asyncio
import re
import websockets

clients = set()
last_y, last_x = 50, 50  # 'hide' 같은 명령에서 사용할 기본 좌표

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

async def handler(ws):
    clients.add(ws)
    try:
        async for msg in ws:
            # 클라이언트->서버 메시지 보이면 로그
            print("from client:", msg)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.remove(ws)

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
        raise ValueError(
            "형식: (v,y,x) | v y x | v,y,x | show y x | hide | q"
        )

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

async def input_loop():
    print(
        "입력 예) (1,50,25) | 1 50 25 | 1,50,25 | show 40 60 | hide | q\n"
        "※ v=0이면 숨김, v=1이면 표시, y/x는 0~100 (퍼센트)\n"
    )
    while True:
        try:
            line = await asyncio.to_thread(input, "좌표 입력> ")
            if not line:
                continue
            v, y, x = parse_line(line)
            pkt = make_packet(v, y, x)
            await send_to_all(pkt)
            print(f"sent: {pkt}  (clients={len(clients)})")
        except SystemExit:
            print("종료합니다.")
            # 연결된 클라이언트에게 마지막으로 숨김 신호 전송(선택)
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
        await input_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
