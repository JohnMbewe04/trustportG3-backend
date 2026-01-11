import os
import io
import json
import re
import typing
import numpy as np
import pdfplumber
import PIL.Image
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIGURATION ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Please add it to your .env file!")

# Configure Google GenAI
genai.configure(api_key=GEMINI_KEY)

# --- APP SETUP ---
app = FastAPI(title="TrustPort Gemini 3 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# PRIVACY GUARD (Same as before)
# ---------------------------------------------------------------------------
class PIISanitizer:
    def sanitize(self, text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_REDACTED]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
        text = re.sub(r'\b\d{10,16}\b', '[ACCOUNT_REDACTED]', text)
        return text

# ---------------------------------------------------------------------------
# GEMINI 3 INTELLIGENCE ENGINE
# ---------------------------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

    def _generate_json(self, prompt: str, image=None) -> dict:
        """Helper to enforce JSON output from Gemini (Replaces OpenAI response_format)"""
        try:
            content = [prompt]
            if image:
                content.append(image)

            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2 # Low temp for factual financial analysis
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini JSON Error: {e}")
            return {}

    def analyze_initial(self, text: str) -> dict:
        print("...Gemini 3 Flash: Running Initial Basel III Analysis...")
        
        prompt = f"""
        You are a senior cross-border banking underwriter using Gemini 3 reasoning capabilities.
        
        Analyze this bank statement text against:
        - Basel III (credit risk, probability of default)
        - IFRS 9 (expected credit loss)
        - FATF (AML red flags)

        Important constraints:
        - DO NOT fabricate specific regulation numbers.
        - Provide an analytical summary.

        Input Text:
        \"\"\"{text[:6000]}\"\"\"

        Output a JSON object with this EXACT schema:
        {{
          "origin_country": "string",
          "currency_symbol": "string",
          "monthly_income": float,
          "savings_rate": float (0-1),
          "risk_flags": int,
          "creditScore": int (300-850),
          "riskProfile": "string (Low/Medium/High)",
          "cashFlow": [float list, 12 months history],
          "predictiveFlow": [float list, 3 months future],
          "transactions": [
            {{
              "date": "YYYY-MM-DD",
              "description": "string",
              "amount": float,
              "is_income": boolean,
              "tag": "string (Verified Income/Recurring/Risk Flag/Other)",
              "is_risky": boolean
            }}
          ]
        }}
        """
        data = self._generate_json(prompt)
        
        # Fallback if JSON fails
        if not data:
            return {
                "origin_country": "Unknown", "creditScore": 600, 
                "riskProfile": "Unknown", "transactions": []
            }
        return data

    def convert_context(self, current_data: dict, target_country: str) -> dict:
        print(f"...Gemini 3 Flash: Converting context to {target_country}...")
        
        prompt = f"""
        Act as a credit risk officer in {target_country}.
        Map this foreign financial profile to local standards.

        Foreign Profile:
        {json.dumps(current_data)}

        Output JSON:
        {{
          "converted_income": int,
          "target_currency_symbol": "string",
          "local_credit_score": int,
          "max_score_in_country": int,
          "analysis_note": "string (1 sentence)",
          "score_explanation": "string (2-3 sentences)",
          "risk_factors": ["string", "string"]
        }}
        """
        return self._generate_json(prompt)

    def generate_interview_question(self, context: dict, chat_history: list) -> str:
        """
        Brain for the Voice Agent (New Feature).
        """
        prompt = f"""
        You are an AI Credit Interviewer. Your goal is to clarify gaps in a user's financial history.
        
        Financial Context:
        Risk Flags: {context.get('risk_flags')}
        Risk Profile: {context.get('riskProfile')}
        Missing Info: {context.get('risk_factors', 'None')}

        Chat History:
        {json.dumps(chat_history)}

        Task: Generate the next single question to ask the user to help improve their trust score.
        If the history is empty, start with a polite greeting and ask about their primary source of income.
        Keep it conversational and short (suitable for Voice TTS).
        """
        # Note: We use standard generate_content (text output) for chat, not JSON
        response = self.model.generate_content(prompt)
        return response.text

    def analyze_vision_fraud(self, file_bytes: bytes):
        """Uses Gemini 3 Flash Vision to detect tampering."""
        try:
            image = PIL.Image.open(io.BytesIO(file_bytes))
            prompt = """
            1. Transcribe all visible text from this document.
            2. FRAUD CHECK: Analyze the image for tampering (font inconsistencies, bad photoshop, pixelation).
            
            Output JSON:
            { "text": "string", "fraud_score": int (0-100) }
            """
            return self._generate_json(prompt, image=image)
        except Exception as e:
            print(f"Vision Error: {e}")
            return {"text": "", "fraud_score": 0}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def extract_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "".join([p.extract_text() or "" for p in pdf.pages])

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

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = (file.filename or "").lower()
        analyzer = GeminiAnalyzer()

        # 1. Extraction & Fraud Check (Vision)
        if filename.endswith(".pdf"):
            raw_text = extract_pdf(contents)
            fraud_score = 0 
        else:
            vision_data = analyzer.analyze_vision_fraud(contents)
            raw_text = vision_data.get("text", "")
            fraud_score = vision_data.get("fraud_score", 0)

        # 2. Sanitize PII
        sanitizer = PIISanitizer()
        safe_text = sanitizer.sanitize(raw_text)

        # 3. Gemini Reasoning Analysis
        ai_data = analyzer.analyze_initial(safe_text)

        # 4. Normalize Data
        core_data = {
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

        return core_data

    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": True, "message": str(e)}


@app.post("/api/convert")
def convert_score(req: ConversionRequest):
    analyzer = GeminiAnalyzer()
    current = req.dict()
    target_country = current.pop("target_country")
    return analyzer.convert_context(current, target_country)


@app.post("/api/generate-letter")
def generate_letter(req: LetterRequest):
    
    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = f"""
    You are an expert in international banking.
    Write a formal tenant/financial verification letter for {req.name}, relocating to {req.country}.
    Their AI-verified TrustPort score is {req.score}.
    
    Requirements:
    - Use a professional tone.
    - Reference Basel III risk concepts (without quoting specific laws).
    - Explain that this score indicates they are a reliable applicant.
    - Include a disclaimer that this is an AI-generated reference.
    """
    response = model.generate_content(prompt)
    return {"letter": response.text}


@app.post("/api/interview/chat")
def interview_chat(req: ChatRequest):
    """
    Brain for the Voice Agent.
    """
    analyzer = GeminiAnalyzer()
    
    new_history = req.history + [{"role": "user", "content": req.message}]
    ai_response_text = analyzer.generate_interview_question(req.context, new_history)
    
    return {
        "reply": ai_response_text,
        "updated_history": new_history + [{"role": "assistant", "content": ai_response_text}]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
