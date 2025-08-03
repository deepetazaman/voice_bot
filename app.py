from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from phq9_session import PHQ9Session
import os

load_dotenv()

app = FastAPI()
client = OpenAI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

phq9_session = PHQ9Session()

class UserInput(BaseModel):
    message: str

@app.get("/")
async def root():
    return FileResponse("static/upload.html")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, file.file, file.content_type)
        )
        return {"transcript": result.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def detect_and_respond(text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly mental health assistant. "
                "If the user seems sad, anxious, or overwhelmed, gently guide them through a PHQ-9 assessment. "
                "Otherwise, chat naturally like a supportive friend."
            )
        },
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

@app.post("/chat")
async def chat(input: UserInput):
    user_text = input.message.strip()

    if user_text.lower() == "start" or phq9_session.started:
        result = phq9_session.process_response(user_text)
        return {
            "reply": result["bot_message"],
            "interrupted": result.get("interrupted", False)
        }

    reply = detect_and_respond(user_text)

    if "PHQ-9" in reply or "Would you like to" in reply or "start a quick screening" in reply:
        intro = "Of course, I'm here to help. The PHQ-9 assessment can help us understand how you’ve been feeling. You can take your time with each response."
        start_msg = phq9_session.start()["bot_message"]
        return {
            "reply": f"{intro}\n\n{start_msg}",
            "interrupted": False
        }

    return {"reply": reply, "interrupted": False}


