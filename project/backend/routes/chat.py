from fastapi import APIRouter
from pydantic import BaseModel
from services import matcher, nlp

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/")
def chat_endpoint(request: ChatRequest):
    user_input = request.message
    
    # Step 1: Extract symptoms (RNN placeholder)
    extracted_symptoms = nlp.extract_symptoms(user_input)
    
    # Step 2: Match to disease
    disease, confidence = matcher.match_symptoms(extracted_symptoms)
    
    # Step 3: Get response
    response = {
        "input": user_input,
        "symptoms": extracted_symptoms,
        "predicted_disease": disease,
        "confidence": confidence,
        "description": matcher.get_description(disease),
        "precautions": matcher.get_precautions(disease)
    }
    return response
