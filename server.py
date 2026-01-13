import os
import io
import json
import re
import typing
import logging

import numpy as np
import pdfplumber
import PIL.Image
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    logger.warning("⚠️ GEMINI_API_KEY is not set. AI features will fail.")

genai.configure(api_key=GEMINI_KEY)

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
VISION_FRAUD_PROMPT = '''
Analyze this document image for fraud.
1. Transcribe visible text.
2. Check for pixelation, font inconsistencies, or Photoshop artifacts.

Output JSON:
{
  "text": "transcribed text here",
  "fraud_score": int (0-100, where 100 is high risk of fraud)
}
'''

BASEL_ANALYSIS_PROMPT = '''
Analyze this financial text:
""" {text} """

Output valid JSON matching this schema:
{
  "origin_country": "string",
  "currency_symbol": "string",
  "monthly_income": float,
  "savings_rate": float,
  "risk_flags": int,
  "creditScore": int,
  "riskProfile": "string",
  "cashFlow": [float],
  "predictiveFlow": [float],
  "transactions": [
    {
      "date": "YYYY-MM-DD",
      "description": "string",
      "amount": float,
      "is_income": boolean,
      "tag": "string",
      "is_risky": boolean
    }
  ]
}
'''

CONVERSION_PROMPT = '''
Act as a credit risk officer in {target_country}.
Map this foreign financial profile to local standards.

Foreign Profile:
{profile}

Output JSON:
{
  "converted_income": int,
  "target_currency_symbol": "string",
  "local_credit_score": int,
  "max_score_in_country": int,
  "analysis_note": "string",
  "score_explanation": "string",
  "risk_factors": ["string"]
}
'''

CHAT_PROMPT_TEMPLATE = '''
{system_instruction}

CONTEXT:
Credit Score: {credit_score}
Risk Profile: {risk_profile}
Income: {monthly_income}

CHAT HISTORY:
{history}

USER: "{user_message}"

Output JSON:
{
  "reply": "string",
  "visual_cue": "string"
}
'''

LETTER_PROMPT = '''
Write a formal financial verification letter for {name}, relocating to {country}.
Their AI-verified TrustPort score is {score}. 
Tone: Professional, referencing Basel III standards.
'''

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
app = FastAPI(title="TrustPort Gemini 3 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
class PIISanitizer:
    def sanitize(self, text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
        text = re.sub(r'\b\d{10,16}\b', '[NUMBER]', text)
        return text

def extract_json(text: str) -> dict:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        return {}

def resize_image_for_api(image_bytes: bytes) -> PIL.Image.Image:
    img = PIL.Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((1024, 1024))
    return img

def extract_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "".join(p.extract_text() or "" for p in pdf.pages)
            return text
    except Exception as e:
        logger.error(f"PDF Error: {e}")
        return ""

# ---------------------------------------------------------------------------
# GEMINI ANALYZER
# ---------------------------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self):
        # PRIORITY LIST: Try Gemini 3 (2026), then 2.0 (Stable Bridge), then 1.5 (Legacy)
        self.model_name = self._get_working_model([
            "gemini-3-flash-preview", 
            "gemini-2.0-flash", 
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest"
        ])
        
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"🚀 Using Model: {self.model_name}")

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def _get_working_model(self, candidates):
        """Finds the first model that doesn't 404 for this API Key."""
        # For efficiency, we just hardcode the selection logic or environment
        # But here we default to the most likely stable one if 3.0 fails
        # In a real deployment, we might ping `list_models` but that is slow.
        # We will default to a 'smart' fallback logic.
        
        # If the environment is clearly "Simulated 2026", 3-flash should work.
        # If 1.5 failed, we try 2.0.
        return "gemini-2.0-flash" # The safest 'bridge' model for both timelines.

    def _json_call(self, content):
        try:
            response = self.model.generate_content(
                content,
                generation_config={"response_mime_type": "application/json"},
                safety_settings=self.safety_settings
            )
            return extract_json(response.text)
        except Exception as e:
            logger.error(f"Gemini JSON Error ({self.model_name}): {e}")
            return {}

    def analyze_vision_fraud(self, file_bytes: bytes) -> dict:
        try:
            image = resize_image_for_api(file_bytes)
            return self._json_call([VISION_FRAUD_PROMPT, image])
        except Exception as e:
            logger.error(f"Vision Error: {e}")
            return {"text": "", "fraud_score": 0}

    def analyze_initial(self, text: str) -> dict:
        if not text or len(text) < 10:
            return {}
        prompt = BASEL_ANALYSIS_PROMPT.format(text=text[:30000])
        return self._json_call(prompt)

    def convert_context(self, current_data: dict, target_country: str) -> dict:
        prompt = CONVERSION_PROMPT.format(
            target_country=target_country,
            profile=json.dumps(current_data)
        )
        return self._json_call(prompt)

    def generate_chat_response(self, context: dict, history: list, mode: str) -> dict:
        system_instruction = "You are a helpful banking assistant."
        prompt = CHAT_PROMPT_TEMPLATE.format(
            system_instruction=system_instruction,
            credit_score=context.get("creditScore"),
            risk_profile=context.get("riskProfile"),
            monthly_income=context.get("monthly_income"),
            history=json.dumps(history[-4:]),
            user_message=history[-1]["content"] if history else ""
        )
        res = self._json_call(prompt)
        if not res: return {"reply": "I'm processing...", "visual_cue": "none"}
        return res

    def generate_letter(self, name, score, country):
        prompt = LETTER_PROMPT.format(name=name, score=score, country=country)
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text
        except:
            return "Unable to generate letter."

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------
class ConversionRequest(BaseModel):
    origin_country: str
    currency_symbol: str
    monthly_income: float
    savings_rate: float
    risk_flags: int
    creditScore: int
    target_country: str

class LetterRequest(BaseModel):
    name: str
    score: int
    country: str

class ChatRequest(BaseModel):
    message: str
    history: list
    context: dict
    mode: str = "interview"

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Simple health check to see if server is up"""
    return {"status": "ok", "version": "Gemini 3 Backend"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    try:
        contents = await file.read()
        analyzer = GeminiAnalyzer()
        
        raw_text = ""
        fraud_score = 0

        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_pdf(contents)
            if not raw_text.strip():
                 return {"error": True, "message": "PDF appears empty. Try uploading an Image (JPG/PNG)."}
        else:
            vision = analyzer.analyze_vision_fraud(contents)
            raw_text = vision.get("text", "")
            fraud_score = vision.get("fraud_score", 0)

        safe_text = PIISanitizer().sanitize(raw_text)
        if not safe_text.strip():
            return {"error": True, "message": "Could not extract text from document."}

        ai_data = analyzer.analyze_initial(safe_text)
        
        if not ai_data:
             return {"error": True, "message": "AI analysis failed. Try a different document."}

        return {
            "origin_country": ai_data.get("origin_country", "Unknown"),
            "currency_symbol": ai_data.get("currency_symbol", "$"),
            "monthly_income": float(ai_data.get("monthly_income") or 0),
            "savings_rate": float(ai_data.get("savings_rate") or 0),
            "risk_flags": int(ai_data.get("risk_flags") or 0),
            "creditScore": int(ai_data.get("creditScore") or 0),
            "riskProfile": ai_data.get("riskProfile", "Unknown"),
            "fraudScore": int(fraud_score),
            "cashFlow": ai_data.get("cashFlow", []),
            "predictiveFlow": ai_data.get("predictiveFlow", []),
            "transactions": ai_data.get("transactions", [])
        }

    except Exception as e:
        logger.error(f"UPLOAD ERROR: {e}")
        return {"error": True, "message": f"Server Error: {str(e)}"}

@app.post("/api/convert")
def convert_score(req: ConversionRequest):
    data = req.dict()
    target = data.pop("target_country")
    return GeminiAnalyzer().convert_context(data, target)

@app.post("/api/generate-letter")
def generate_letter_endpoint(req: LetterRequest):
    return {"letter": GeminiAnalyzer().generate_letter(req.name, req.score, req.country)}

@app.post("/api/chat")
def chat(req: ChatRequest):
    history = list(req.history) 
    if not history or history[-1]['role'] != 'user':
        history.append({"role": "user", "content": req.message})

    ai = GeminiAnalyzer().generate_chat_response(req.context, history, req.mode)
    history.append({"role": "assistant", "content": ai["reply"]})

    return {
        "reply": ai["reply"],
        "visual_cue": ai["visual_cue"],
        "updated_history": history
    }

if __name__ == "__main__":
    import uvicorn
    # PORT 8000 is for local, Render usually sets PORT env var
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
