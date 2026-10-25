"""
Cara pakai:
1. Di notebook, ambil beberapa baris data mentah (13 kolom) beserta probabilitas
   ensemble hasil perhitungan notebook, simpan ke tests/referensi.json:
   [{"input": {...13 kolom...}, "prob": 0.4321}, ...]
2. Jalankan server: uvicorn api.main:app
3. Jalankan: python tests/compare.py
"""
import json

import requests

URL = "http://127.0.0.1:8000/predict"

with open("tests/referensi.json", encoding="utf-8") as f:
    data = json.load(f)

maks = 0.0
for row in data:
    r = requests.post(URL, json=row["input"], timeout=30)
    r.raise_for_status()
    selisih = abs(r.json()["probability"] - row["prob"])
    maks = max(maks, selisih)
    print(f"notebook={row['prob']:.5f}  api={r.json()['probability']:.5f}  selisih={selisih:.5f}")

print("\nSelisih terbesar:", maks)
print("LOLOS" if maks < 1e-3 else "BEDA, periksa preprocessing / prior_scale")