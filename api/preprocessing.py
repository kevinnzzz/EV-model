import pandas as pd

FREQ_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CAT_COLS = [
    "Age", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

NUMERIC_COLS = FREQ_COLS + ["Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]

# ===== ISI DARI main.ipynb (nilai di bawah hanya contoh) =====
INCOME_CLIP = (30000.0, 200000.0)
COMMUTE_CLIP = (5.0, 80.0)
# =============================================================


def make_features(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    """Ubah 13 kolom mentah menjadi 35 kolom seperti saat training."""
    df = df.copy()

    # Data training kemungkinan bertipe float (kunci freq berbentuk "36.0"),
    # jadi samakan agar pencarian tabel dan teks _cat konsisten.
    for c in NUMERIC_COLS:
        df[c] = df[c].astype(float)

    # --- Fitur turunan: SAMAKAN rumusnya dengan main.ipynb ---
    df["Total_Charging_Stations"] = (
        df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )
    df["Income_per_Car"] = df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"] + 1)
    df["Concern_x_Subsidy"] = df["Environmental_Concern_Level"] * df["Subsidy_Available"]
    df["Income_x_Concern"] = df["Annual_Income_USD"] * df["Environmental_Concern_Level"]
    df["Anxiety_x_Commute"] = df["Range_Anxiety_Level"] * df["Daily_Commute_km"]
    df["Commute_per_Station"] = df["Daily_Commute_km"] / (df["Total_Charging_Stations"] + 1)
    df["Charging_Access"] = df["Home_Charging_Possible"] + df["Total_Charging_Stations"]  # TEBAKAN

    # --- Fitur di-clip ---
    df["Income_Clipped"] = df["Annual_Income_USD"].clip(*INCOME_CLIP)
    df["Commute_Clipped"] = df["Daily_Commute_km"].clip(*COMMUTE_CLIP)

    # --- Fitur frekuensi (nilai yang belum pernah terlihat -> 0) ---
    for c in FREQ_COLS:
        df[f"{c}_freq"] = df[c].map(freq_maps[c]).fillna(0.0)

    # --- Fitur kategori (teks dari angka) ---
    for c in CAT_COLS:
        df[f"{c}_cat"] = df[c].astype(str)

    return df