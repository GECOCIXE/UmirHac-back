import os, io, base64, asyncio, time
from time import localtime, strftime
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# ---------- app / logging ----------
app = FastAPI()
try:
    from rich import print as rprint
    RICH = True
except Exception:
    RICH = False

def log(text: str, color: str = "yellow", fmt: str = "[bold {color}]{time}[/] => {text}"):
    ts = strftime("%H:%M:%S", localtime())
    s = fmt.format(time=ts, text=text, color=color)
    if RICH:
        rprint(s)
    else:
        print(s.replace("[/]", "").replace(f"[bold {color}]", ""))

# ---------- schemas ----------
class TxtReq(BaseModel):
    prompt: str

class EditReq(BaseModel):
    prompt: str
    image_base64: str  # PNG/JPEG в base64

# ---------- globals ----------
# один общий семафор → максимум 1 инференс на GPU одновременно
_gpu_lock = asyncio.Semaphore(1)

# ---------- health ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": False,  # Заглушка - всегда возвращает False
        "torch": "stub",  # Заглушка - возвращает строку
    }

# ---------- t2i: schnell ----------
@app.post("/generate_image")
async def generate_image(req: TxtReq):
    # Валидация данных
    if not req.prompt or len(req.prompt.strip()) == 0:
        log("Empty prompt received", "red")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    h = int(os.getenv("HEIGHT", "512"))
    w = int(os.getenv("WIDTH",  "512"))
    steps = int(os.getenv("STEPS", "3"))
    gscale = float(os.getenv("SCHNELL_GUIDANCE", "0.0"))
    log(f"schnell request: {h}x{w}, steps={steps}, guidance={gscale}")
    t0 = time.perf_counter()

    # Имитация работы сервера
    try:
        async with _gpu_lock:
            # Здесь будет имитация работы модели
            # Создаем заглушку изображения
            import PIL
            from PIL import Image
            img = Image.new('RGB', (w, h), color='red')  # Заглушка - красное изображение
            
    except Exception as e:
        log(f"schnell error: {e}", "red")
        raise HTTPException(status_code=500, detail=f"schnell failed: {e}")

    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    log(f"schnell done in {time.perf_counter()-t0:.2f}s")
    return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

# ---------- edit: kontext ----------
@app.post("/edit_image")
async def edit_image(req: EditReq):
    # Валидация данных
    if not req.prompt or len(req.prompt.strip()) == 0:
        log("Empty prompt received", "red")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if not req.image_base64:
        log("Empty image_base64 received", "red")
        raise HTTPException(status_code=400, detail="image_base64 cannot be empty")

    # входная картинка
    try:
        raw = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad image_base64: {e}")

    W, H = image.size

    # параметры «неон без каши»
    steps  = int(os.getenv("KONTEXT_STEPS",  "22"))       # 20–24
    gscale = float(os.getenv("KONTEXT_GUIDANCE", "4.0"))  # 3.8–5.0 (ниже ~3.5 бывает «чёрный»)
    seed   = int(os.getenv("SEED", "42"))

    # negative_prompt опционален (по умолчанию выключен)
    neg_env = os.getenv("KONTEXT_NEGATIVE", "").strip()
    negative = None if neg_env == "" else neg_env

    # управление размером:
    # keep  — работать и вернуть в исходном размере
    # native — прогнать в «родном» размере модели и вернуть его
    # upscale_then_keep — прогнать на native_max, вернуть исходный размер (↑чёткость)
    mode = os.getenv("KONTEXT_RESIZE_MODE", "upscale_then_keep")  # keep | native | upscale_then_keep
    native_max = int(os.getenv("KONTEXT_NATIVE_MAX", "768"))

    if mode == "keep":
        tgt_w, tgt_h = W, H
        in_img = image
        return_size = (W, H)
    else:
        scale = native_max / max(W, H)
        tgt_w, tgt_h = _round8(W * scale), _round8(H * scale)
        if mode == "upscale_then_keep" and scale > 1.0:
            in_img = image.resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)
            return_size = (W, H)
        else:
            in_img = image
            return_size = (tgt_w, tgt_h)

    log(f"kontext: in={W}x{H} mode={mode} -> run={tgt_w}x{tgt_h} return={return_size}, steps={steps}, g={gscale}, neg={'on' if negative else 'off'}")
    t0 = time.perf_counter()

    try:
        async with _gpu_lock:
            # Здесь будет имитация работы модели
            # Создаем заглушку изображения на основе входного
            out = image.resize(return_size, Image.Resampling.LANCZOS)  # Просто изменяем размер как заглушка
            
    except Exception as e:
        log(f"kontext error: {e}", "red")
        raise HTTPException(status_code=500, detail=f"kontext failed: {e}")

    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)
    log(f"kontext done in {time.perf_counter()-t0:.2f}s")
    return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

# ---------- helpers ----------
def _round8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3339)
