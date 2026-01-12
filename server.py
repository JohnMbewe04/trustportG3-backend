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
    raise RuntimeError("GEMINI_API_KEY is not set.")

genai.configure(api_key=GEMINI_KEY)

# ---------------------------------------------------------------------------
# PROMPT CONSTANTS (FULL LOGIC RESTORED)
# ---------------------------------------------------------------------------

VISION_FRAUD_PROMPT = '''
1. Transcribe all visible text from this document.
2. FRAUD CHECK: Analyze the image for tampering (font inconsistencies, photoshop artifacts, pixelation).

Output JSON:
{
  "text": "string",
  "fraud_score": int
}
'''

# ✅ RESTORED: Specific calculation instructions (Steps 1-10) added back
BASEL_ANALYSIS_PROMPT = '''
You are a senior cross-border banking underwriter.

Analyze the following bank statement text against:
- Basel III (credit risk, probability of default)
- IFRS 9 (expected credit loss)
- FATF (AML red flags)

Input Text:
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

Output JSON (EXACT schema):
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

CHAT_SYSTEM_PERSONAS = {
    "interview": (
        "You are a strict but fair Credit Underwriter. "
        "Ask probing questions about income stability and risk flags found in the data."
    ),
    "informative": (
        "You are a Financial Literacy Teacher. "
        "Explain the user's credit profile and Basel III concepts in simple terms."
    ),
    "analyst": (
        "You are a Forensic Document Analyst. "
        "Discuss formatting inconsistencies, metadata, or specific transaction patterns."
    ),
    "default": "You are a helpful banking assistant."
}

CHAT_PROMPT_TEMPLATE = '''
{system_instruction}

CONTEXT:
- Credit Score: {credit_score}
- Risk Profile: {risk_profile}
- Monthly Income: {monthly_income}
- Risk Flags: {risk_flags}

CHAT HISTORY:
{history}

USER MESSAGE:
"{user_message}"

TASK:
1. Respond conversationally to the user.
2. Select a visual cue to update the dashboard from:
   ['score_gauge', 'cashflow_chart', 'risk_list', 'income_card', 'none']

OUTPUT JSON ONLY:
{
  "reply": "string",
  "visual_cue": "string"
}
'''

LETTER_PROMPT = '''
You are an expert in international banking.

Write a formal financial verification letter for {name}, relocating to {country}.
Their AI-verified TrustPort score is {score}.

Requirements:
- Professional tone.
- Reference Basel III risk concepts (do not quote laws).
- Explain that this score indicates reliability.
- Include a disclaimer that this is AI-generated.
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
# PRIVACY GUARD
# ---------------------------------------------------------------------------
class PIISanitizer:
    def sanitize(self, text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
        text = re.sub(r'\b\d{10,16}\b', '[NUMBER]', text)
        return text

# ---------------------------------------------------------------------------
# GEMINI ANALYZER
# ---------------------------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

    def _json_call(self, content):
        try:
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini JSON Error: {e}")
            return {}

    def analyze_vision_fraud(self, file_bytes: bytes) -> dict:
        try:
            image = PIL.Image.open(io.BytesIO(file_bytes))
            return self._json_call([VISION_FRAUD_PROMPT, image])
        except Exception as e:
            print(f"Vision Error: {e}")
            return {"text": "", "fraud_score": 0}

    def analyze_initial(self, text: str) -> dict:
        prompt = BASEL_ANALYSIS_PROMPT.format(text=text[:8000])
        data = self._json_call(prompt)
        # Fallback if empty
        if not data:
             return {"origin_country": "Unknown", "creditScore": 0, "transactions": []}
        return data

    def convert_context(self, current_data: dict, target_country: str) -> dict:
        prompt = CONVERSION_PROMPT.format(
            target_country=target_country,
            profile=json.dumps(current_data)
        )
        return self._json_call(prompt)

    def generate_chat_response(self, context: dict, history: list, mode: str) -> dict:
        system_instruction = CHAT_SYSTEM_PERSONAS.get(mode, CHAT_SYSTEM_PERSONAS["default"])

        prompt = CHAT_PROMPT_TEMPLATE.format(
            system_instruction=system_instruction,
            credit_score=context.get("creditScore"),
            risk_profile=context.get("riskProfile"),
            monthly_income=context.get("monthly_income"),
            risk_flags=context.get("risk_flags"),
            history=json.dumps(history[-6:]),
            user_message=history[-1]["content"] if history else ""
        )

        result = self._json_call(prompt)
        if not result:
            return {"reply": "I'm having trouble analyzing that right now.", "visual_cue": "none"}
        return result

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def extract_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        return ""

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
    try:
        contents = await file.read()
        analyzer = GeminiAnalyzer()

        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_pdf(contents)
            fraud_score = 0
        else:
            vision = analyzer.analyze_vision_fraud(contents)
            raw_text = vision.get("text", "")
            fraud_score = vision.get("fraud_score", 0)

        safe_text = PIISanitizer().sanitize(raw_text)
        ai_data = analyzer.analyze_initial(safe_text)

        return {
            "origin_country": ai_data.get("origin_country", "Unknown"),
            "currency_symbol": ai_data.get("currency_symbol", "$"),
            "monthly_income": float(ai_data.get("monthly_income", 0)),
            "savings_rate": float(ai_data.get("savings_rate", 0)),
            "risk_flags": int(ai_data.get("risk_flags", 0)),
            "creditScore": int(ai_data.get("creditScore", 0)),
            "riskProfile": ai_data.get("riskProfile", "Unknown"),
            "fraudScore": fraud_score,
            "cashFlow": ai_data.get("cashFlow", []),
            "predictiveFlow": ai_data.get("predictiveFlow", []),
            "transactions": ai_data.get("transactions", [])
        }
    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": True, "message": str(e)}

@app.post("/api/convert")
def convert_score(req: ConversionRequest):
    analyzer = GeminiAnalyzer()
    data = req.dict()
    target = data.pop("target_country")
    return analyzer.convert_context(data, target)

@app.post("/api/generate-letter")
def generate_letter(req: LetterRequest):
    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = LETTER_PROMPT.format(
        name=req.name,
        country=req.country,
        score=req.score
    )
    response = model.generate_content(prompt)
    return {"letter": response.text}

@app.post("/api/chat")
def chat(req: ChatRequest):
    analyzer = GeminiAnalyzer()

    # Avoid modifying the request object in place
    history = list(req.history) 
    
    # Add User Message if not present (simple check)
    if not history or history[-1]['role'] != 'user':
        history.append({"role": "user", "content": req.message})

    ai = analyzer.generate_chat_response(req.context, history, req.mode)
    
    # Add AI Response
    history.append({"role": "assistant", "content": ai["reply"]})

    return {
        "reply": ai["reply"],
        "visual_cue": ai["visual_cue"],
        "updated_history": history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
