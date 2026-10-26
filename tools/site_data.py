"""Данные для страницы библиотеки на сайте daruma.audio.

Запускается в GitHub Actions (.github/workflows/clap.yml) перед подсчётом векторов CLAP.
Пишет:
  previews/<id>.mp3     — лёгкая копия для прослушивания (MP3 128 кбит/с); делается только для новых
                          и изменившихся звуков (сверка по previews/manifest.json);
  embeddings/catalog.json — все звуки: id, путь, пак, название, описание, папки, длительность, волна, размер,
                          точки синхронизации start / sync / end (мс) и тип kind (hit / whoosh / texture).
Исходные WAV не меняются; скачиваются на сайте прямо с GitHub.
"""
import csv
import glob
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

PACKS = {"PHANTOM": "Phantom Sound Pack", "DARUMA VOL. 01": "Daruma Vol. 01"}
BARS = 48  # столбиков в волне


def sid(path):
    s = path.lower().replace("daruma vol. 01", "vol01")
    return re.sub(r"[^a-z0-9]+", "-", s.rsplit(".", 1)[0]).strip("-")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Точки для синхронизации с видео (мс от начала файла). Файлы не обрезаны: у многих в начале тишина,
# а у вжухов главный момент («пролёт») наступает через полсекунды и позже.
#   start — где звук реально начинается (выше −40 дБ от пика);
#   sync  — что совмещать с событием в видео: у ударов и кликов — атака самого громкого удара,
#           у вжухов — громкий момент пролёта, у длинных фактур (письмо, печать, часы) — начало;
#   end   — где затихает хвост (ниже −60 дБ от пика);
#   kind  — hit / whoosh / texture.
TEXTURE_CATS = {"WRITE", "CLOCK"}
TEXTURE_WORDS = ("TYPE", "DATA_TXT", "WRITE", "CLOCK", "TICK", "LONG_SLIDE")


def timing(path, cat, name):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = np.abs(a).max(axis=1)
    hop = max(1, int(sr * 0.005))                       # 5 мс
    n = len(a) // hop
    if n < 2:
        return {"start": 0, "sync": 0, "end": int(len(a) / sr * 1000), "kind": "hit"}
    env = a[: n * hop].reshape(n, hop).max(axis=1)      # огибающая по пикам
    top = env.max() or 1.0
    db = 20 * np.log10(np.maximum(env, 1e-9) / top)
    start = int(np.argmax(db > -40))
    end = int(n - 1 - np.argmax(db[::-1] > -60))
    up = name.upper()
    active = db[start:end + 1]
    loud_share = float((active > -20).mean()) if len(active) else 0
    long_ms = (end - start) * 5
    if cat in TEXTURE_CATS or any(w in up for w in TEXTURE_WORDS) or (long_ms > 2500 and loud_share > 0.3):
        kind, sync = "texture", start
    elif cat == "WHOOSH" or "WHOOSH" in up:
        # пролёт — максимум сглаженной (≈50 мс) громкости
        k = 10
        smooth = np.convolve(env ** 2, np.ones(k) / k, mode="same")
        kind, sync = "whoosh", int(np.argmax(smooth))
    else:
        # удар — первый громкий удар (не тише −6 дБ от пика), отступаем к началу его атаки (−20 дБ от него);
        # если атака дольше 80 мс, это нарастание (появление, подъём) — синхронизация, как у вжуха, по громкому моменту
        f = int(np.argmax(env >= top * 0.5))
        local = f
        while local + 1 < n and env[local + 1] >= env[local]:
            local += 1
        j = f
        while j > start and env[j - 1] > env[local] * 0.1:
            j -= 1
        if (local - j) * 5 > 80:
            k = 10
            smooth = np.convolve(env ** 2, np.ones(k) / k, mode="same")
            kind, sync = "whoosh", int(np.argmax(smooth))
        else:
            kind, sync = "hit", j
    return {"start": start * 5, "sync": sync * 5, "end": end * 5, "kind": kind}


def peaks(path):
    a, _ = sf.read(path, dtype="float32", always_2d=True)
    a = np.abs(a).max(axis=1)
    if not len(a):
        return "0" * BARS
    parts = np.array_split(a, BARS)
    v = np.array([p.max() if len(p) else 0 for p in parts])
    v = v / (v.max() or 1)
    return "".join("0123456789abcdef"[int(round(x * 15))] for x in v)


os.makedirs("previews", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
man_path = "previews/manifest.json"
manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
new_manifest, items, made = {}, [], 0

for meta in sorted(glob.glob("*/metadata.csv")):
    pack = os.path.dirname(meta)
    for row in csv.DictReader(open(meta, encoding="utf-8")):
        file = f"{pack}/{row['file']}"
        i = sid(file)
        h = md5(file)
        mp3 = f"previews/{i}.mp3"
        if manifest.get(i) != h or not os.path.exists(mp3):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", file, "-codec:a", "libmp3lame",
                            "-b:a", "128k", "-ac", "2", mp3], check=True)
            made += 1
        new_manifest[i] = h
        items.append({
            "id": i,
            "file": file,
            "pack": pack,
            "name": row["name"],
            "desc": row.get("description", ""),
            "cat": row.get("category", ""),
            "sub": row.get("subcategory", ""),
            "dur": float(row.get("duration_s") or 0),
            "peaks": peaks(file),
            **timing(file, row.get("category", ""), row["name"]),
            "bytes": os.path.getsize(file),
        })

# копии звуков, которых больше нет в библиотеке
for old in set(manifest) - set(new_manifest):
    p = f"previews/{old}.mp3"
    if os.path.exists(p):
        os.remove(p)

json.dump(new_manifest, open(man_path, "w"), indent=0, sort_keys=True)
json.dump({"packs": PACKS, "items": items}, open("embeddings/catalog.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(f"каталог: {len(items)} звуков, новых копий для прослушивания: {made}")
