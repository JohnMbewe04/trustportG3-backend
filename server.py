import os
import io
import json
import re
import typing

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
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    # Use a dummy key locally to prevent immediate crash, but warn user
    print("WARNING: GEMINI_API_KEY is not set.")

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
You are a senior cross-border banking underwriter (Basel III expert).

Analyze this financial text:
""" {text} """

TASK:
1. Detect ORIGIN COUNTRY and CURRENCY.
2. Estimate MONTHLY INCOME (average of regular salary/credit entries).
3. Estimate SAVINGS RATE (0-1).
4. Count RISK FLAGS (gambling, crypto, overdrafts).
5. Compute CREDIT SCORE (300-850) based on stability.
6. Assign RISK PROFILE (Low/Medium/High).
7. Construct CASH FLOW history (12 months).
8. Construct PREDICTIVE flow (next 3 months).
9. Extract 30 REPRESENTATIVE transactions.

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
  "visual_cue": "string" (one of: 'score_gauge', 'cashflow_chart', 'risk_list', 'income_card', 'none')
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
        # Simple regex for emails and phone numbers
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
        text = re.sub(r'\b\d{10,16}\b', '[NUMBER]', text)
        return text

def extract_json(text: str) -> dict:
    """Robust JSON extraction using Regex"""
    try:
        # Find JSON object between braces
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text) # Fallback
    except Exception:
        return {}

def resize_image_for_api(image_bytes: bytes) -> PIL.Image.Image:
    """Resize huge images to prevent timeouts/payload errors"""
    img = PIL.Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if larger than 1024x1024 to save latency
    img.thumbnail((1024, 1024)) 
    return img

def extract_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "".join(p.extract_text() or "" for p in pdf.pages)
            return text
    except Exception as e:
        print(f"PDF Error: {e}")
        return ""

# ---------------------------------------------------------------------------
# GEMINI ANALYZER
# ---------------------------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self):
        # Using Gemini 1.5 Flash as it is stable for vision/long context
        # Adjust to "gemini-3-flash-preview" if you have specific access
        self.model = genai.GenerativeModel("gemini-1.5-flash") 

        # Turn off safety filters for financial doc processing (prevents false positives)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def _json_call(self, content):
        try:
            response = self.model.generate_content(
                content,
                generation_config={"response_mime_type": "application/json"},
                safety_settings=self.safety_settings
            )
            
            # Check if response was blocked
            if not response.parts:
                print(f"Gemini Blocked Response: {response.prompt_feedback}")
                return {}

            return extract_json(response.text)
            
        except Exception as e:
            print(f"Gemini JSON Error: {e}")
            return {}

    def analyze_vision_fraud(self, file_bytes: bytes) -> dict:
        try:
            image = resize_image_for_api(file_bytes)
            return self._json_call([VISION_FRAUD_PROMPT, image])
        except Exception as e:
            print(f"Vision Error: {e}")
            return {"text": "", "fraud_score": 0}

    def analyze_initial(self, text: str) -> dict:
        if not text or len(text) < 10:
            return {} # Empty text, cannot analyze
            
        prompt = BASEL_ANALYSIS_PROMPT.format(text=text[:30000]) # 1.5 Flash has large context
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
        result = self._json_call(prompt)
        # Default fallback
        if not result:
            return {"reply": "I'm processing that information.", "visual_cue": "none"}
        return result

    def generate_letter(self, name, score, country):
        prompt = LETTER_PROMPT.format(name=name, score=score, country=country)
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text
        except:
            return "Unable to generate letter at this time."

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
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    try:
        contents = await file.read()
        analyzer = GeminiAnalyzer()
        
        raw_text = ""
        fraud_score = 0

        # 1. Handle PDF
        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_pdf(contents)
            # Fallback: If PDF text is empty, it might be scanned.
            # For hackathon simple fix: If empty, warn user.
            if not raw_text.strip():
                 return {"error": True, "message": "Uploaded PDF appears to be empty or scanned images. Please upload a JPG/PNG of the statement instead."}
        
        # 2. Handle Image
        else:
            vision = analyzer.analyze_vision_fraud(contents)
            raw_text = vision.get("text", "")
            fraud_score = vision.get("fraud_score", 0)

        # 3. Sanitize & Analyze
        safe_text = PIISanitizer().sanitize(raw_text)
        
        # If still empty after vision
        if not safe_text.strip():
            return {"error": True, "message": "Could not extract text from document."}

        ai_data = analyzer.analyze_initial(safe_text)
        
        # If AI failed to return JSON
        if not ai_data:
             return {"error": True, "message": "AI analysis failed. Please try a clearer document."}

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
        print(f"CRITICAL UPLOAD ERROR: {e}")
        return {"error": True, "message": f"Server Error: {str(e)}"}

@app.post("/api/convert")
def convert_score(req: ConversionRequest):
    analyzer = GeminiAnalyzer()
    data = req.dict()
    target = data.pop("target_country")
    return analyzer.convert_context(data, target)

@app.post("/api/generate-letter")
def generate_letter_endpoint(req: LetterRequest):
    analyzer = GeminiAnalyzer()
    return {"letter": analyzer.generate_letter(req.name, req.score, req.country)}

@app.post("/api/chat")
def chat(req: ChatRequest):
    analyzer = GeminiAnalyzer()
    history = list(req.history) 
    if not history or history[-1]['role'] != 'user':
        history.append({"role": "user", "content": req.message})

    ai = analyzer.generate_chat_response(req.context, history, req.mode)
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
