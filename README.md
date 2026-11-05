# Smart Signal Alert (Python + Lightweight Charts)

คำนวณเงื่อนไขเดียวกับอินดิเคเตอร์: **Status 3 ตัว (Trend BB, Trend MA, Kernel @1H) ไปทางเดียวกัน + Smart Buy/Sell**
แล้วแจ้งเตือน Telegram และแสดงกราฟบนเว็บ

## ตั้งค่า (ครั้งเดียว)
1. สร้างบอทกับ @BotFather -> ได้ Bot Token; ส่งข้อความหาบอท 1 ครั้ง แล้วหา Chat ID (เช่น @userinfobot)
2. สร้าง GitHub repo (public ถ้าจะใช้ Pages ฟรี) แล้วอัปโหลดโฟลเดอร์ `smart_alert/`, `docs/`, `.github/`
3. Repo -> Settings -> Secrets and variables -> Actions: เพิ่ม `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
4. Repo -> Settings -> Pages: Source = Deploy from branch, branch main, folder /docs
5. แท็บ Actions -> smart-alert -> Run workflow (ทดสอบครั้งแรก)

หน้าเว็บ: https://<user>.github.io/<repo>/

## ปรับค่า
แก้ `smart_alert/config.json` (symbol, timeframe, balance, risk, SL buffer ฯลฯ)

## ทดสอบในเครื่อง
    pip install -r smart_alert/requirements.txt
    cd smart_alert && python run.py --demo   # ข้อมูลจำลอง ไม่ต่อเน็ต

## ข้อจำกัด
- GitHub Actions cron อย่างต่ำ 5 นาที และอาจหน่วง -> ไม่ใช่ real-time เป๊ะ (ราคาบนกราฟ live ผ่าน WebSocket แต่สัญญาณอัปเดตตามรอบ cron)
- ราคา PAXGUSDT ไม่เท่า XAUUSD เป๊ะ; ผลอาจต่างจาก TradingView เล็กน้อย
- Port เฉพาะ EMA/SMA/RMA และ Kernel แบบ Rational Quadratic (ค่าตั้งต้นของสคริปต์)
- ไม่รวม Orderblocks, S/R, Smart Trail (ไม่ได้ใช้ในเงื่อนไข alert)
