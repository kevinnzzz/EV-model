import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# Alamat website yang boleh memanggil API ini, pisahkan dengan koma.
# Contoh di server: ALLOWED_ORIGINS=https://ev-web.vercel.app
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]