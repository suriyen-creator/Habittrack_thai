# AI Habit Tracker (Pandas + scikit-learn)

แปลงบันทึกกิจวัตรรายวัน (ข้อความไทย/อังกฤษ) → ตารางฟีเจอร์ → Insight รายสัปดาห์
รุ่น: baseline v0.1 · อัปเดต: 2025-09-15

---

## 1) ความเข้ากันได้ (Compatibility)

* **Python:** แนะนำ **3.11**, รองรับ **3.10–3.12**

  > หมายเหตุ: Python 3.13 บางแพลตฟอร์มอาจยังไม่มี wheel ของ `scikit-learn`
* **OS:** Windows 10/11, macOS 12+, Linux (Ubuntu 20.04+)
* **สถาปัตยกรรม:** x86\_64 (ใช้งานได้บน ARM/M1/M2 แต่อาจติดตั้งช้ากว่า)

## 2) Dependencies หลัก

* `pandas>=2.1,<3.0`
* `numpy>=1.26,<3.0`
* `scikit-learn>=1.4,<2.0`
  ตัวเลือก: `matplotlib` (ทำกราฟ), `streamlit` (ทำแดชบอร์ด)

## 3) โครงสร้างโปรเจ็กต์ (แนะนำ)

```
.
├─ Habittrack.py          # โค้ดหลัก (รวมแพตช์ is_weekend)
├─ requirements.txt       # รายการไลบรารีหลัก
└─ your_logs.csv          # (เลือกใส่) บันทึกกิจวัตรของคุณ
```

## 4) การติดตั้ง (Setup)

### วิธี A) ใช้ pip

```bash
python -m venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
# Windows (cmd):        .\venv\Scripts\activate
# macOS/Linux:          source venv/bin/activate
pip install -r requirements.txt
```

### วิธี B) ใช้ uv (เร็วและแคชดี)

```bash
uv venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
# macOS/Linux:          source venv/bin/activate
uv pip install -r requirements.txt
```

**PowerShell tip:** ถ้าขึ้น *running scripts is disabled* ให้รัน:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 5) การใช้งาน (Usage)

### 5.1 โหมดเดโม่

```bash
python Habittrack.py
```

* ถ้าไม่พบ `your_logs.csv` จะใช้ข้อมูลตัวอย่างอัตโนมัติ
* ผลลัพธ์: แสดงตารางฟีเจอร์ช่วงท้าย ๆ, สรุป **INSIGHTS (7 วัน)**, และบันทึก `daily_features.csv`

### 5.2 ใช้กับข้อมูลจริง

1. สร้าง `your_logs.csv` (UTF-8) ตามสคีมาในหัวข้อถัดไป
2. รัน `python Habittrack.py`
3. ดูสรุปในคอนโซล + ไฟล์ `daily_features.csv`

**ตัวอย่าง `your_logs.csv`:**

```csv
date,note,wellbeing
2025-08-17,"นอน 6 ชม. เดิน 6500 ก้าว กาแฟ 2 แก้ว มือถือ 2 ชม. ดื่มน้ำ 1200 ml วิ่ง 20 นาที",72
2025-08-18,"sleep 5.5 hr, coffee 1 cup, steps 5200, water 1.5 L, screen 120 min",65
2025-08-19,"นอน 7 ชม. เดิน 8000 ก้าว ออกกำลังกาย 30 นาที น้ำ 1800 ml",80
```

## 6) สคีมาข้อมูลอินพุต (Input Schema)

* **คอลัมน์บังคับ:**

  * `date` (รูปแบบ YYYY-MM-DD)
  * `note` (ข้อความไทย/อังกฤษ; ควรมีหน่วยชัดเจน เช่น hr, min, ml, L, cup)
* **คอลัมน์เสริม:**

  * `wellbeing` (ต่อเนื่อง 0–100 หรือฉลาก good/ok/bad)

**ตัวแปรที่สกัดจาก `note` (ถ้าพบ):**

* `sleep_hours` (ชั่วโมงนอน: “นอน 6 ชม.”, “sleep 7 hr”)
* `steps` (จำนวนก้าว: “เดิน 8000 ก้าว”, “steps 6500”)
* `exercise_min` (นาทีออกกำลัง: “ออกกำลังกาย 30 นาที”)
* `coffee_cups` (แก้วกาแฟ/วัน; สะสมได้ถ้าพบหลายครั้ง)
* `water_ml` (มล.น้ำดื่ม; รองรับ “1.5 L” → 1500 ml)
* `screen_time_min` (นาทีจอ; รองรับ “1.5 h” → 90 นาที)
* `alcohol_units` (ดริ๊งค์แอลกอฮอล์/วัน; สะสมได้)
* `junk_food` (ธง 0/1 ถ้าพบคำ: ฟาสต์ฟู้ด/ของทอด/ของหวาน/junk)

## 7) หลักการทำงาน (Under the Hood)

1. **Normalize ข้อความ:** แปลงเลขไทย→อารบิก, lower case, เก็บสัญลักษณ์ที่จำเป็น (.: % / + -)
2. **Regex ไทย/อังกฤษ:** จับ pattern สำหรับ sleep/steps/exercise/coffee/water/screen/alcohol/junk
3. **Aggregate รายวัน:** `groupby(date)` → ใช้ `max` หรือ `sum` เหมาะกับชนิดข้อมูล + เติมค่าเริ่มต้น
4. **Feature Engineering:**

   * rolling 7 วัน: `*_roll7_mean`, `*_roll7_delta`
   * `streak_exercise` (ตั้งเกณฑ์ออกกำลัง ≥20 นาที/วัน)
   * `sleep_deficit_7d` (เทียบเป้า 7 ชม./คืน)
   * **`is_weekend`** ใช้ `.dt.dayofweek >= 5` (แพตช์แทน `.weekday`)
5. **Insight (Rule-based):**

   * ✅ จุดเด่น: เช่น สตรีคออกกำลัง ≥3 วัน, นอนเฉลี่ย ≥7 ชม., คาเฟอีนเฉลี่ย ≤1 แก้ว
   * ⚠️ ระวัง: เช่น นอน <6 ชม. 2 วันติด, กิจกรรมต่ำกว่าค่าเฉลี่ยหลายวัน, น้ำดื่ม <1500 ml
   * 🎯 แนะนำ: เพิ่มเวลานอน, เดิน ≥7k ก้าว/วัน, ออกกำลัง ≥150 นาที/สัปดาห์, ดื่มน้ำ \~1.8–2.2 L/วัน
6. **โมเดลพื้นฐาน (ถ้ามี `wellbeing`):**

   * ค่าต่อเนื่อง → `LinearRegression` (รายงาน R² + อันดับฟีเจอร์สำคัญ)
   * ฉลาก → `LogisticRegression` (รายงาน Accuracy)

## 8) การตั้งค่า/ปรับแต่ง (Config)

* ปรับเกณฑ์ Insight ใน `summarize_insights()`
* เปลี่ยนเกณฑ์สตรีคใน `exercise_flag` (ดีฟอลต์ ≥20 นาที/วัน)
* เป้าหมายการนอน: ปรับสูตร `sleep_deficit_7d` (ดีฟอลต์ 7 ชม./คืน)
* เพิ่มคำพ้อง/หน่วย: แก้ลิสต์ `PATTERNS`

## 9) Troubleshooting

* **`AttributeError: 'Series' object has no attribute 'weekday'`**
  ใช้ `d["date"] = pd.to_datetime(...); d["date"].dt.dayofweek >= 5` (แก้แล้วในไฟล์ให้เรียบร้อย)
* **PowerShell: running scripts is disabled**
  รัน:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
* **ติดตั้ง `scikit-learn` ไม่ผ่านบน Python 3.13**
  ใช้ Python 3.10–3.12 แล้วติดตั้งใหม่
* **จับค่าจากข้อความไม่ขึ้น**
  ตรวจว่ามีคำ/หน่วยตาม regex (hr, min, ml, L, cup) หรือเพิ่ม pattern เอง

## 10) ข้อจำกัด & คำเตือน

* เป็นข้อมูลเชิงสุขภาพทั่วไป **ไม่ใช่คำแนะนำทางการแพทย์**
* การสกัดข้อมูลแบบ rule-based อาจพลาดสแลง/พิมพ์ผิด
* ค่าที่ไม่ระบุในวันนั้นจะถูกเติม 0 — ปรับตามบริบทของคุณ

## 11) Roadmap (สั้น ๆ)

* Streamlit dashboard (metric cards + กราฟ rolling 7 วัน)
* แยก intent ข้อความด้วย TF-IDF/Classifier
* รองรับข้อมูลจากอุปกรณ์สวมใส่ (steps/HR/HRV)
* โมเดลเชิงเวลา + เป้าหมายส่วนบุคคล (personalized goals/alerts)

## 12) License

MIT-like (เพื่อการเรียนรู้/งานส่วนตัว). ช่วยให้เครดิตเมื่อเผยแพร่สาธารณะจะน่ารักมาก 🙌

---

ถ้าอยากได้เวอร์ชันภาษาอังกฤษ/มีภาพตัวอย่างกราฟ/แดชบอร์ด บอกได้เลย เดี๋ยวจัดให้ทันที!
