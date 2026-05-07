from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import pytesseract
from PIL import Image
import sqlite3
from datetime import datetime
import platform

# Import your AI model testing function
from test_meme import test_single_meme 

# ==========================================
# ⚙️ Tesseract OCR Configuration
# ==========================================
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# 🗄️ Database Initialization (First time only)
# ==========================================
def init_db():
    conn = sqlite3.connect('meme_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_name TEXT,
                  text_used TEXT,
                  prediction TEXT,
                  confidence REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.get("/")
def read_root():
    return {"message": "Sinhala Meme Hate Speech API is Running! 🚀"}

@app.post("/predict")
async def predict_meme(file: UploadFile = File(...), text: str = Form("")): 
    try:
        # 1. Save the uploaded image
        file_extension = file.filename.split(".")[-1]
        new_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 🔍 Extract Text (Via OCR or User Input)
        final_text = text.strip()
        
        if not final_text:
            print("⏳ No text provided by user. Extracting text via OCR...")
            try:
                img = Image.open(file_path)
                img_gray = img.convert('L')
                extracted_text = pytesseract.image_to_string(img_gray, lang='sin+eng').strip()
                
                if extracted_text:
                    final_text = extracted_text
                    print(f"✅ Extracted text via OCR: {final_text}")
                else:
                    final_text = " "
                    print("⚠️ No text found by OCR.")
            except Exception as e:
                print(f"❌ OCR Error: {e}")
                final_text = " "
        else:
            print(f"✅ Using user-provided text: {final_text}")

        # 3. 🚀 Send data to the AI Model
        ai_response = test_single_meme(file_path, final_text) 
        
        # 4. 🗄️ Insert prediction data into the Database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect('meme_database.db')
        c = conn.cursor()
        c.execute("INSERT INTO history (image_name, text_used, prediction, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (new_filename, final_text, ai_response["result"], ai_response["confidence"], timestamp))
        conn.commit()
        conn.close()

        # 5. Return the final JSON response
        return {
            "status": "success",
            "prediction": ai_response["result"],         
            "confidence": ai_response["confidence"],     
            "image_saved_at": file_path,
            "input_text": final_text
        }
    except Exception as e:
        # 🚨 මෙන්න අලුතින් දැම්ම කෑල්ල: Error එකේ මුල ඉඳන් අගටම Terminal එකේ ප්‍රින්ට් වෙනවා
        import traceback
        print("\n🚨 FATAL ERROR IN PREDICTION:")
        traceback.print_exc() 
        return {"status": "error", "message": str(e)}

# ==========================================
# 📊 Endpoint to send data to the Admin Dashboard
# ==========================================
@app.get("/dashboard-data")
def get_dashboard_data():
    conn = sqlite3.connect('meme_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    history_list = []
    hateful_count = 0
    non_hateful_count = 0

    for row in rows:
        history_list.append({
            "id": row[0],
            "image_name": row[1],
            "text": row[2],
            "prediction": row[3],
            "confidence": row[4],
            "time": row[5]
        })
        if row[3] == "HATEFUL":
            hateful_count += 1
        else:
            non_hateful_count += 1

    return {
        "total_checked": len(rows),
        "hateful_total": hateful_count,
        "non_hateful_total": non_hateful_count,
        "history": history_list
    }

# ==========================================
# 🗑️ Endpoint to clear history from Database
# ==========================================
@app.delete("/clear-history")
def clear_history():
    try:
        conn = sqlite3.connect('meme_database.db')
        c = conn.cursor()
        c.execute("DELETE FROM history") 
        conn.commit()
        conn.close()
        return {"status": "success", "message": "History cleared successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 🗑️ Delete a single record by ID
# ==========================================
@app.delete("/delete-history/{record_id}")
def delete_single_record(record_id: int):
    try:
        conn = sqlite3.connect('meme_database.db')
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Record {record_id} deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}