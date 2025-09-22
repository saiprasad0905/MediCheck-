from fastapi import FastAPI
from routes import chat

app = FastAPI(
    title="Symptom Checker Chatbot",
    description="An AI-powered chatbot that predicts diseases based on symptoms.",
    version="1.0.0"
)

# Include chat routes
app.include_router(chat.router)

# Root check
@app.get("/")
def root():
    return {"message": "Symptom Checker Chatbot API is running 🚀"}
