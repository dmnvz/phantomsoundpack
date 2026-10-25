"""Векторы CLAP для всех звуков библиотеки.

Запускается в GitHub Actions (.github/workflows/clap.yml).
Пишет в embeddings/:
  audio.json  — вектор каждого звука (int8, base64), для поиска и «похожих звуков»;
  vocab.json  — векторы слов из описаний (для подсказки тегов);
  report.md   — пробные запросы, похожие звуки, подсказанные теги для звуков без описаний.
"""
import base64
import csv
import glob
import json
import os
import re

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

MODEL = os.environ.get("CLAP_MODEL", "laion/larger_clap_general")
SR = 48000
OUT = "embeddings"

torch.set_num_threads(os.cpu_count() or 2)
model = ClapModel.from_pretrained(MODEL).eval()
proc = ClapProcessor.from_pretrained(MODEL)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        import resampy
        a = resampy.resample(a, sr, SR)
    return a


def audio_emb(paths, bs=8):
    out = []
    for i in range(0, len(paths), bs):
        batch = [load(p) for p in paths[i:i + bs]]
        inp = proc(audios=batch, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            out.append(model.get_audio_features(**inp).numpy())
        print(f"звуки: {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    return norm(np.concatenate(out))


def text_emb(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        inp = proc(text=texts[i:i + bs], return_tensors="pt", padding=True)
        with torch.no_grad():
            out.append(model.get_text_features(**inp).numpy())
    return norm(np.concatenate(out))


def pack_int8(m):
    q = np.clip(np.round(m * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode()


# --- звуки и описания ---
items = []
for meta in sorted(glob.glob("*/metadata.csv")):
    pack = os.path.dirname(meta)
    for row in csv.DictReader(open(meta, encoding="utf-8")):
        items.append({
            "file": f"{pack}/{row['file']}",
            "pack": pack,
            "name": row["name"],
            "description": row.get("description", ""),
            "category": row.get("category", ""),
            "subcategory": row.get("subcategory", ""),
            "duration": float(row.get("duration_s") or 0),
        })

A = audio_emb([it["file"] for it in items])
os.makedirs(OUT, exist_ok=True)
json.dump({
    "model": MODEL, "dim": int(A.shape[1]), "dtype": "int8", "scale": 127,
    "items": [{k: it[k] for k in ("file", "pack", "name", "category", "subcategory")} for it in items],
    "vectors": pack_int8(A),
}, open(f"{OUT}/audio.json", "w"), ensure_ascii=False)

# --- словарь тегов из описаний ---
words = sorted({w.strip().lower() for it in items for w in it["description"].split(",") if w.strip()})
V = text_emb([f"the sound of {w}" for w in words]) if words else np.zeros((0, A.shape[1]))
json.dump({"model": MODEL, "dim": int(A.shape[1]), "dtype": "int8", "scale": 127,
           "words": words, "vectors": pack_int8(V)}, open(f"{OUT}/vocab.json", "w"), ensure_ascii=False)

# --- отчёт ---
queries = [l.strip() for l in open("tools/test_queries.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
Q = text_emb(queries)
lines = [f"# Проверка CLAP\n\nМодель: `{MODEL}`. Звуков: {len(items)}. Слов в словаре тегов: {len(words)}.\n",
         "## Поиск по фразе (лучшие 5)\n"]
for q, qv in zip(queries, Q):
    s = A @ qv
    top = np.argsort(-s)[:5]
    lines.append(f"**{q}**\n")
    lines += [f"- {s[i]:.2f} `{items[i]['file']}`" for i in top]
    lines.append("")

lines.append("## Похожие звуки (лучшие 4)\n")
rng = np.random.default_rng(1)
for i in sorted(rng.choice(len(items), size=min(12, len(items)), replace=False)):
    s = A @ A[i]
    top = [j for j in np.argsort(-s) if j != i][:4]
    lines.append(f"**{items[i]['file']}** → " + ", ".join(f"`{items[j]['name']}` {s[j]:.2f}" for j in top))
lines.append("")

if len(words):
    lines.append("## Подсказанные теги для звуков без описания\n")
    for i, it in enumerate(items):
        if it["description"]:
            continue
        s = V @ A[i]
        lines.append(f"- `{it['file']}`: " + ", ".join(words[j] for j in np.argsort(-s)[:6]))

open(f"{OUT}/report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("готово:", len(items), "звуков,", len(words), "слов")
