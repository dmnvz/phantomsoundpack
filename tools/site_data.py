"""Данные для страницы библиотеки на сайте daruma.audio.

Запускается в GitHub Actions (.github/workflows/clap.yml) перед подсчётом векторов CLAP.
Пишет:
  previews/<id>.mp3     — лёгкая копия для прослушивания (MP3 128 кбит/с); делается только для новых
                          и изменившихся звуков (сверка по previews/manifest.json);
  embeddings/catalog.json — все звуки: id, путь, пак, название, описание, папки, длительность, волна, размер.
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
