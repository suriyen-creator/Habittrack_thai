# habit_tracker_baseline.py
import re
import math
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---------- 1) Utilities ----------
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

def norm_text(t: str) -> str:
    if pd.isna(t):
        return ""
    t = t.strip().lower().translate(THAI_DIGITS)
    t = re.sub(r"[^\w\s\.:%/+-]", " ", t)  # เก็บ ., :, %, /, +, -
    t = re.sub(r"\s+", " ", t)
    return t

# ---------- 2) Rule patterns (ไทย/อังกฤษ) ----------
PATTERNS = [
    # sleep
    (r"(นอน|หลับ|sleep)\s*(\d+(\.\d+)?)\s*(ชม\.|ชั่วโมง|h|hr|hrs?)", "sleep_hours"),
    # steps
    (r"(เดิน|ก้าว|steps?)\s*(\d{3,6})", "steps"),
    # exercise minutes
    (r"(วิ่ง|ออกกำลังกาย|exercise|workout|ปั่น|โยคะ|วิดพื้น)[^\d]*(\d+)\s*(นาที|mins?|m)", "exercise_min"),
    # coffee cups
    (r"(กาแฟ|coffee|เอสเปรสโซ่|ลาเต้)[^\d]*(\d+)\s*(แก้ว|cups?)", "coffee_cups"),
    # water (ml or liters)
    (r"(น้ำ|water)[^\d]*(\d+)\s*ml", "water_ml"),
    (r"(น้ำ|water)[^\d]*(\d+(\.\d+)?)\s*l", "water_l"),
    # screen time
    (r"(จอ|มือถือ|หน้าจอ|screen)[^\d]*(\d+)\s*(นาที|mins?|m)", "screen_time_min"),
    (r"(จอ|มือถือ|หน้าจอ|screen)[^\d]*(\d+(\.\d+)?)\s*(ชม\.|ชั่วโมง|h|hr)", "screen_time_hr"),
    # alcohol
    (r"(เบียร์|ไวน์|เหล้า|alcohol|beer|wine)[^\d]*(\d+)\s*(แก้ว|ดริ๊งค์|drinks?)", "alcohol_units"),
    # junk food (binary cues)
    (r"(ฟาสต์ฟู้ด|ของทอด|ของหวาน|ขนม|junk|soda|โซดาหวาน)", "junk_food_flag"),
]

def parse_note(note: str) -> dict:
    t = norm_text(note)
    out = {
        "sleep_hours": np.nan,
        "steps": np.nan,
        "exercise_min": np.nan,
        "coffee_cups": 0.0,
        "water_ml": np.nan,
        "screen_time_min": np.nan,
        "alcohol_units": 0.0,
        "junk_food": 0,
    }
    for pat, key in PATTERNS:
        for m in re.finditer(pat, t):
            if key == "sleep_hours":
                out["sleep_hours"] = float(m.group(2))
            elif key == "steps":
                out["steps"] = float(m.group(2))
            elif key == "exercise_min":
                out["exercise_min"] = float(m.group(2))
            elif key == "coffee_cups":
                out["coffee_cups"] += float(m.group(2))
            elif key == "water_ml":
                out["water_ml"] = float(m.group(2))
            elif key == "water_l":
                out["water_ml"] = float(m.group(2)) * 1000
            elif key == "screen_time_min":
                out["screen_time_min"] = float(m.group(2))
            elif key == "screen_time_hr":
                out["screen_time_min"] = float(m.group(2)) * 60
            elif key == "alcohol_units":
                out["alcohol_units"] += float(m.group(2))
            elif key == "junk_food_flag":
                out["junk_food"] = 1
    return out

# ---------- 3) Build daily table ----------
def build_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    parsed = df["note"].apply(parse_note).apply(pd.Series)
    daily = pd.concat([df[["date"]], parsed], axis=1).groupby("date", as_index=False).agg({
        "sleep_hours":"max", "steps":"max", "exercise_min":"sum", "coffee_cups":"sum",
        "water_ml":"max", "screen_time_min":"max", "alcohol_units":"sum", "junk_food":"max"
    })
    # fill reasonable defaults
    daily["water_ml"] = daily["water_ml"].fillna(0)
    daily["sleep_hours"] = daily["sleep_hours"].fillna(0)
    daily["exercise_min"] = daily["exercise_min"].fillna(0)
    daily["steps"] = daily["steps"].fillna(0)
    daily["screen_time_min"] = daily["screen_time_min"].fillna(0)
    daily["alcohol_units"] = daily["alcohol_units"].fillna(0)
    daily["junk_food"] = daily["junk_food"].fillna(0)
    # features
    d = daily.sort_values("date").reset_index(drop=True)
    for col in ["sleep_hours","steps","exercise_min","coffee_cups","water_ml","screen_time_min"]:
        d[f"{col}_roll7_mean"] = d[col].rolling(7, min_periods=1).mean()
        d[f"{col}_roll7_delta"] = d[col] - d[f"{col}_roll7_mean"]
    # streak exercise
    d["exercise_flag"] = (d["exercise_min"]>=20).astype(int)
    d["streak_exercise"] = d["exercise_flag"] * (d["exercise_flag"].groupby((d["exercise_flag"]!=d["exercise_flag"].shift()).cumsum()).cumcount()+1)
    # sleep deficit 7d
    d["sleep_deficit_7d"] = (7*7) - d["sleep_hours"].rolling(7, min_periods=1).sum()  # สมมติ 7 ชม./คืน เป็นเป้าหมาย
    d["is_weekend"] = pd.to_datetime(d["date"], errors="coerce").dt.dayofweek >= 5# หรือ .dt.weekday ก็ได้เหมือนกัน
    return d

# ---------- 4) Insight rules ----------
def summarize_insights(d: pd.DataFrame, n_days=7) -> str:
    week = d.tail(n_days)
    msgs = []
    # ✅ จุดเด่น
    highlights = []
    if (week["streak_exercise"].max() >= 3):
        highlights.append(f"ออกกำลังกายสตรีค {int(week['streak_exercise'].max())} วัน 🎉")
    if (week["sleep_hours"].mean() >= 7):
        highlights.append("ค่าเฉลี่ยการนอน ≥ 7 ชม. ดีต่อการฟื้นตัว")
    if (week["coffee_cups"].mean() <= 1):
        highlights.append("คาเฟอีนเฉลี่ยต่ำ (≤1 แก้ว/วัน)")
    if highlights:
        msgs.append("✅ จุดเด่น: " + " · ".join(highlights))
    # ⚠️ ระวัง
    warns = []
    if (week["sleep_hours"] < 6).rolling(2).sum().max() >= 2:
        warns.append("นอนน้อยกว่า 6 ชม. 2 วันติด")
    if (week["steps"] < week["steps_roll7_mean"]*0.8).sum() >= 2:
        warns.append("กิจกรรมต่ำกว่าค่าเฉลี่ยหลายวัน")
    if (week["coffee_cups"] > week["coffee_cups_roll7_mean"]+1).sum() >= 1:
        warns.append("คาเฟอีนสูงกว่าปกติบางวัน")
    if (week["water_ml"] < 1500).sum() >= 3:
        warns.append("ดื่มน้ำน้อย (<1500 ml) หลายวัน")
    if warns:
        msgs.append("⚠️ ควรระวัง: " + " · ".join(warns))
    # 🎯 ข้อเสนอแนะ
    tips = []
    if week["sleep_deficit_7d"].iloc[-1] > 0:
        tips.append("เพิ่มเวลานอนให้เฉลี่ย ≥ 7 ชม./คืน")
    if (week["steps"].mean() < 7000):
        tips.append("ตั้งเป้าเดิน ≥ 7,000 ก้าว/วัน")
    if (week["exercise_min"].sum() < 150):
        tips.append("สะสมออกกำลังกาย ≥ 150 นาที/สัปดาห์")
    if (week["water_ml"].mean() < 1800):
        tips.append("ดื่มน้ำให้ถึง ~1.8–2.2 ลิตร/วัน")
    if tips:
        msgs.append("🎯 แนะนำถัดไป: " + " · ".join(tips))
    return "\n".join(msgs) if msgs else "ยังไม่มี insight เด่นชัดในสัปดาห์นี้"

# ---------- 5) โมเดล baseline (ถ้ามี target) ----------
def train_baseline(daily_with_target: pd.DataFrame):
    df = daily_with_target.dropna(subset=["wellbeing"]).copy()
    y = df["wellbeing"]
    X = df.drop(columns=["date","wellbeing"])
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    if y.nunique() > 3 and y.dtype != "object":
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print("R2:", round(r2_score(y_test, pred), 3))
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=lambda s: s.abs(), ascending=False)
        print("Top features:\n", coefs.head(10))
    else:
        # ถ้า wellbeing เป็น label: good/ok/bad (แปลงเป็น 0/1/2 ก่อน)
        y_enc = y.astype("category").cat.codes
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_enc.loc[X_train.index])
        acc = model.score(X_test, y_enc.loc[X_test.index])
        print("Accuracy:", round(acc,3))

# ---------- 6) Demo (ตัวอย่างข้อมูลเล็ก ๆ) ----------
if __name__ == "__main__":
    data = [
        {"date":"2025-08-17","note":"นอน 6 ชม. เดิน 6500 ก้าว กาแฟ 2 แก้ว มือถือ 2 ชม. ดื่มน้ำ 1200 ml วิ่ง 20 นาที"},
        {"date":"2025-08-18","note":"sleep 5.5 hr, coffee 1 cup, steps 5200, water 1.5 L, screen 120 min"},
        {"date":"2025-08-19","note":"นอน 7 ชม. เดิน 8000 ก้าว ออกกำลังกาย 30 นาที น้ำ 1800 ml"},
        {"date":"2025-08-20","note":"นอน 5 ชม. กาแฟ 3 แก้ว เบียร์ 2 แก้ว เดิน 4000 ก้าว"},
        {"date":"2025-08-21","note":"นอน 7.5 ชม. เดิน 9000 ก้าว น้ำ 2 L ออกกำลัง 25 นาที"},
        {"date":"2025-08-22","note":"sleep 7 hr, steps 7000, coffee 1 cup, screen 1.5 h"},
        {"date":"2025-08-23","note":"นอน 6 ชม. เดิน 5000 ก้าว ของทอด"},
    ]
    raw = pd.read_csv("test.csv")
    daily = build_daily(raw)
    print(daily.tail(3))
    print("\nINSIGHTS (7d):\n", summarize_insights(daily, n_days=7))