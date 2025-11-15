import os, io, gc, base64, asyncio, time
from time import localtime, strftime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFilter
import torch

from diffusers import FluxPipeline, FluxKontextPipeline  # требуется diffusers>=0.35

# ---------- allocator tweaks ----------
# новая переменная у PyTorch (оставим и старую на всякий случай)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

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
_pipe_schnell: Optional[FluxPipeline] = None
_pipe_kontext: Optional[FluxKontextPipeline] = None

# один общий семафор → максимум 1 инференс на GPU одновременно
_gpu_lock = asyncio.Semaphore(1)

# ---------- helpers ----------
def _dtype_for_device() -> torch.dtype:
    # V100 не умеет bfloat16 → fp16 на GPU, fp32 на CPU
    return torch.float16 if torch.cuda.is_available() else torch.float32

def _enable_memory_savers(pipe):
    try:
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    except Exception as e:
        log(f"memory opts skipped: {e}", "red")

def _round8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)

def _post_sharpen(img: Image.Image) -> Image.Image:
    if os.getenv("KONTEXT_SHARPEN", "0") != "1":
        return img
    radius = float(os.getenv("SHARP_RADIUS", "1.2"))
    amount = int(os.getenv("SHARP_AMOUNT", "120"))
    thresh = int(os.getenv("SHARP_THRESHOLD", "3"))
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=thresh))

# ---------- startup ----------
@app.on_event("startup")
def load_models():
    global _pipe_schnell, _pipe_kontext
    if _pipe_schnell and _pipe_kontext:
        return

    use_cuda = torch.cuda.is_available()
    dtype = _dtype_for_device()

    id_schnell = os.getenv("MODEL_SCHNELL", "black-forest-labs/FLUX.1-schnell")
    id_kontext = os.getenv("MODEL_KONTEXT", "black-forest-labs/FLUX.1-Kontext-dev")

    # Бюджеты VRAM/RAM (для device_map='balanced')
    # Для V100 32GB типично 17/13 ГиБ с запасом под буферы.
    s_cuda = os.getenv("SCHNELL_CUDA_GB", "17GiB")
    k_cuda = os.getenv("KONTEXT_CUDA_GB", "13GiB")
    cpu_mem = os.getenv("CPU_MAX_GB", "10GiB")
    offload = os.getenv("OFFLOAD_DIR", "/tmp/umirhack_offload")
    os.makedirs(offload, exist_ok=True)

    # ---- SCHNELL ----
    log(f"Loading {id_schnell} (dtype={dtype}, device_map=balanced VRAM={s_cuda})")
    _pipe_schnell = FluxPipeline.from_pretrained(
        id_schnell,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map="balanced",  # у FLUX поддержаны 'balanced' и 'cuda'
        max_memory={0: s_cuda, "cpu": cpu_mem},
        offload_folder=os.path.join(offload, "schnell"),
    )
    _enable_memory_savers(_pipe_schnell)

    # ---- KONTEXT ----
    log(f"Loading {id_kontext} (dtype={dtype}, device_map=balanced VRAM={k_cuda})")
    _pipe_kontext = FluxKontextPipeline.from_pretrained(
        id_kontext,
        torch_dtype=dtype,  # на V100 — fp16
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map="balanced",
        max_memory={0: k_cuda, "cpu": cpu_mem},
        offload_folder=os.path.join(offload, "kontext"),
    )
    _enable_memory_savers(_pipe_kontext)

    # warmup для снижения пиков на первом реальном запросе
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    try:
        with torch.inference_mode():
            _ = _pipe_schnell("warmup", height=256, width=256, num_inference_steps=1, guidance_scale=0.0).images[0]
    except Exception as e:
        log(f"warmup skipped: {e}", "red")

    log("Both models are ready")

# ---------- health ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "torch": torch.__version__,
    }

# ---------- t2i: schnell ----------
@app.post("/generate_image")
async def generate_image(req: TxtReq):
    assert _pipe_schnell is not None, "schnell not loaded"

    h = int(os.getenv("HEIGHT", "512"))
    w = int(os.getenv("WIDTH",  "512"))
    steps = int(os.getenv("STEPS", "3"))
    gscale = float(os.getenv("SCHNELL_GUIDANCE", "0.0"))
    log(f"schnell request: {h}x{w}, steps={steps}, guidance={gscale}")
    t0 = time.perf_counter()

    try:
        async with _gpu_lock:
            with torch.inference_mode():
                img = _pipe_schnell(
                    req.prompt, height=h, width=w, num_inference_steps=steps, guidance_scale=gscale
                ).images[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    except Exception as e:
        log(f"schnell error: {e}", "red")
        raise HTTPException(status_code=500, detail=f"schnell failed: {e}")

    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    log(f"schnell done in {time.perf_counter()-t0:.2f}s")
    return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

# ---------- edit: kontext ----------
@app.post("/edit_image")
async def edit_image(req: EditReq):
    assert _pipe_kontext is not None, "kontext not loaded"

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

    kwargs = dict(
        image=in_img,
        prompt=req.prompt,
        guidance_scale=gscale,
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),  # CPU-генератор ок
        output_type="pil",
        width=tgt_w, height=tgt_h,                      # фиксируем размер прогона
    )
    if negative:
        kwargs["negative_prompt"] = negative

    log(f"kontext: in={W}x{H} mode={mode} -> run={tgt_w}x{tgt_h} return={return_size}, steps={steps}, g={gscale}, neg={'on' if negative else 'off'}")
    t0 = time.perf_counter()

    try:
        async with _gpu_lock:
            with torch.inference_mode():
                out = _pipe_kontext(**kwargs).images[0]
                if return_size != (out.width, out.height):
                    out = out.resize(return_size, Image.Resampling.LANCZOS)
                out = _post_sharpen(out)  # опциональный шейпинг
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    except Exception as e:
        log(f"kontext error: {e}", "red")
        raise HTTPException(status_code=500, detail=f"kontext failed: {e}")

    buf = io.BytesIO(); out.save(buf, "PNG"); buf.seek(0)
    log(f"kontext done in {time.perf_counter()-t0:.2f}s")
    return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}