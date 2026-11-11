"""SenseCore GPU self-check: everything v6 training and VLM work will touch, on one GPU.

1. torch / CUDA / bf16 matmul on this card.
2. LLM2Vec (Llama-3-8B + both LoRA adapters) encodes a HumanoidArena double-desk instruction,
   written as a text-embedding cache file.
3. The released v1 HumanoidArena double-desk checkpoint (ft_419h, SONIC) answers one /infer request
   built from a blank image and a standing-still state history.
4. Qwen3-VL-2B-Instruct (candidate VLM) points at a red square in a synthetic image; latency is
   timed, since the plan assumes 150-400 ms per VLM call.
Prints one PASS/FAIL line per step and exits non-zero if any step failed.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

H = Path(os.environ["KVLA_HOME"])
WEIGHTS = Path(os.environ["KVLA_WEIGHTS"])
CACHE = H / "cache" / "arena_text"
RESULTS: list[tuple[str, bool, str]] = []


def step(name):
    def wrap(fn):
        t0 = time.time()
        try:
            note = fn() or ""
            RESULTS.append((name, True, f"{note} ({time.time() - t0:.0f}s)"))
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            traceback.print_exc()
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"[{'PASS' if RESULTS[-1][1] else 'FAIL'}] {name}: {RESULTS[-1][2]}", flush=True)
        torch.cuda.empty_cache()
        return fn
    return wrap


@step("cuda")
def _():
    assert torch.cuda.is_available(), "no CUDA device"
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    (a @ a).float().sum().item()
    return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, {torch.cuda.device_count()} GPU(s)"


def double_desk_instruction() -> str:
    import pandas as pd

    path = next(iter(glob.glob(str(Path(os.environ["KVLA_ARENA_DATA"]) / "HOI_double_desk/sonic_refpose_v3_1/meta/tasks.parquet"))))
    df = pd.read_parquet(path)
    return str(df["task"].iloc[0] if "task" in df.columns else df.index[0]).strip()


@step("llm2vec text cache")
def _():
    from kimodo_vla.model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder

    instruction = double_desk_instruction()
    encoder = LLM2VecEncoder(checkpoint_path=str(WEIGHTS / "checkpoints")).to("cuda").eval()
    with torch.no_grad():
        emb, _ = encoder([instruction])  # [1, 1, 4096]
    emb = emb.reshape(1, -1).to(torch.bfloat16).cpu()
    assert emb.shape == (1, 4096) and torch.isfinite(emb.float()).all(), emb.shape
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 3, "source": "humanoidarena", "task_id": "doubledesk", "task_name": "HOI_double_desk",
                "instruction": instruction, "embedding": emb}, CACHE / "doubledesk.pt")
    del encoder
    return f"'{instruction[:60]}' -> {tuple(emb.shape)}"


@step("v1 arena double-desk inference")
def _():
    from kimodo_vla.benchmarks.arena.policy import make_kimodo_policy

    ckpt = sorted(glob.glob(str(WEIGHTS / "HumanoidArena_Single_Task/ft_419h_*HOI_double_desk_sonic/*/checkpoint_200000")))[-1]
    policy = make_kimodo_policy(ckpt, str(CACHE), controller="sonic", device="cuda")
    policy.reset(2)
    image = np.zeros((480, 640, 3), np.uint8)
    state = np.zeros((100, 64), np.float32)
    state[:, 0], state[:, 4] = 1.0, 1.0  # identity rot6d, joints at zero, at rest
    request = {"task": "doubledesk", "observation": {
        "images": {"front": {"shape": list(image.shape), "dtype": "uint8", "data_b64": base64.b64encode(image.tobytes()).decode()}},
        "state_history": state.tolist()}}
    times = []
    for _ in range(3):
        t0 = time.time(); reply = policy(request); torch.cuda.synchronize(); times.append(time.time() - t0)
    arrays = {k: np.asarray(v).shape for k, v in reply.items() if isinstance(v, list)}
    assert arrays, f"no array in reply: {list(reply)}"
    for k, v in reply.items():
        if isinstance(v, list):
            assert np.isfinite(np.asarray(v, dtype=np.float64)).all(), f"{k} not finite"
    return f"reply {arrays}, {1000 * min(times):.0f} ms per plan"


@step("qwen3-vl-2b pointing")
def _():
    from PIL import Image, ImageDraw
    from transformers import AutoModelForImageTextToText, AutoProcessor

    path = os.environ["KVLA_VLM"]
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    img = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(img).rectangle([400, 300, 460, 360], fill="red")  # centre (430, 330)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Point to the red square. Answer only with JSON {\"point_2d\": [x, y]}."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to("cuda")
    times = []
    for _ in range(3):
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize(); times.append(time.time() - t0)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    mem = torch.cuda.max_memory_allocated() / 2**30
    return f"answer {text!r} (target ~ (430,330) px or ~ (672,688) on a 0-1000 grid), {1000 * min(times):.0f} ms, peak {mem:.1f} GiB"


failed = [n for n, ok, _ in RESULTS if not ok]
print(json.dumps({"passed": [n for n, ok, _ in RESULTS if ok], "failed": failed}), flush=True)
sys.exit(1 if failed else 0)
