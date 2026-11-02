import os
import random
import numpy as np
from aiohttp import web
from stable_baselines3 import PPO

# 1. 서버 시작 시 체크포인트 모델들을 메모리에 미리 로드
MODELS = {}
MODEL_NAMES = [
    "stork_ai_65536_steps",
    "stork_ai_131072_steps",
    "stork_ai_204800_steps",
    "ppo_300k"
]

print("📦 AI 체크포인트 모델들을 메모리에 로드 중입니다...")
for name in MODEL_NAMES:
    for path in [f"./checkpoints/{name}.zip", f"./{name}.zip"]:
        if os.path.exists(path):
            MODELS[name] = PPO.load(path, device="cpu")
            print(f"  ✅ 로드 성공: {name}")
            break

DEFAULT_MODEL = next(iter(MODELS.keys())) if MODELS else None
user_selected_model = {}

# 2. 난이도 변경 및 리셋 제어 소켓 (/ws/control)
async def control_ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    client_ip = request.remote
    if client_ip not in user_selected_model:
        user_selected_model[client_ip] = DEFAULT_MODEL

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = msg.data
            if data.startswith("MODEL:"):
                model_name = data.split(":")[1]
                if model_name in MODELS:
                    user_selected_model[client_ip] = model_name
                    await ws.send_str(f"LOADED:{model_name}")
            elif data == "RESET":
                pass

    return ws

# 3. Ruffle AI 게임 물리 통신 소켓 (/ws/game)
async def game_ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    client_ip = request.remote
    delay_counter = 0
    current_action = 0
    buffer = b""

    async for msg in ws:
        if msg.type in (web.WSMsgType.BINARY, web.WSMsgType.TEXT):
            data = msg.data.encode("utf-8") if isinstance(msg.data, str) else msg.data
            buffer += data

            while b"\x00" in buffer:
                raw_msg, buffer = buffer.split(b"\x00", 1)
                text_msg = raw_msg.decode("utf-8", errors="ignore").strip()
                if not text_msg:
                    continue

                try:
                    parts = [float(x) for x in text_msg.split(",")]
                    obs = np.array(parts[:4], dtype=np.float32)

                    model_name = user_selected_model.get(client_ip, DEFAULT_MODEL)
                    model = MODELS.get(model_name)

                    if delay_counter <= 0:
                        if model is not None:
                            action, _ = model.predict(obs, deterministic=False)
                            current_action = int(action)
                        else:
                            current_action = 0
                        delay_counter = random.randint(3, 6)
                    else:
                        delay_counter -= 1

                    send_action = current_action if delay_counter >= 3 else 0
                    await ws.send_bytes(f"{send_action}\x00".encode("utf-8"))

                except ValueError:
                    pass

    return ws

async def index_handler(request):
    return web.FileResponse("./index.html")

app = web.Application()
app.router.add_get("/", index_handler)
app.router.add_get("/ws/control", control_ws_handler)
app.router.add_get("/ws/game", game_ws_handler)
app.router.add_static("/", path="./", show_index=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"=== 🌐 통합 웹 서버 시작: http://localhost:{port} ===")
    web.run_app(app, host="0.0.0.0", port=port)
