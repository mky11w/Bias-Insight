import os
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, Float, String, JSON, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")


sentiment_model, sentiment_tokenizer = load_model("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_model, emotion_tokenizer = load_model("j-hartmann/emotion-english-distilroberta-base")
political_model, political_tokenizer = load_model("premsa/political-bias-prediction-allsides-BERT")
stereo_model, stereo_tokenizer = load_model("wu981526092/Sentence-Level-Stereotype-Detector")


emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
stereo_labels = ["unrelated", "stereotype_gender", "stereotype_race", "stereotype_profession", "stereotype_religion"]
political_labels = ["left", "center", "right"]


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"
    id = Column(Integer, primary_key=True, index=True)
    website = Column(String, index=True)
    entry_time = Column(Float)
    exit_time = Column(Float, nullable=True)
    viewing_time = Column(Float, default=0.0)
    sentiment_score = Column(Float)
    emotions = Column(JSON)
    political_bias = Column(JSON)
    stereotype_score = Column(JSON)  

Base.metadata.create_all(bind=engine)


class TextInput(BaseModel):
    texts: List[str]
    website: str
    entry_time: float

class TimeUpdate(BaseModel):
    website: str
    exit_time: float


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def classify_texts(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().tolist()  # Convert tensors to lists


@app.post("/SMBSanalyze")
async def analyze_texts(input_data: TextInput, db: Session = Depends(get_db)):
    texts = input_data.texts
    if not texts or all(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Input texts cannot be empty.")


    sentiment_scores = await classify_texts(sentiment_model, sentiment_tokenizer, texts)
    batch_sentiment_score = round(sum(score[1] - score[0] for score in sentiment_scores) / len(sentiment_scores), 4)


    emotion_scores = await classify_texts(emotion_model, emotion_tokenizer, texts)
    batch_emotions = {label: round(sum(em[i] for em in emotion_scores) / len(emotion_scores), 4) for i, label in enumerate(emotion_labels)}

    political_scores = await classify_texts(political_model, political_tokenizer, texts)
    batch_political_bias = {label: round(sum(pol[i] for pol in political_scores) / len(political_scores), 4) for i, label in enumerate(political_labels)}


    stereotype_scores = await classify_texts(stereo_model, stereo_tokenizer, texts)
    batch_stereotypes = {label: round(sum(st[i] for st in stereotype_scores) / len(stereotype_scores), 4) for i, label in enumerate(stereo_labels)}


    response = {
        "sentiment_score": batch_sentiment_score,
        "dominating_emotions": batch_emotions,
        "political_bias": batch_political_bias,
        "stereotype_analysis": batch_stereotypes
    }

    session = AnalysisSession(
        website=input_data.website,
        entry_time=input_data.entry_time,
        sentiment_score=batch_sentiment_score,  
        emotions=batch_emotions if batch_emotions else {}, 
        political_bias=batch_political_bias if batch_political_bias else {}, 
        stereotype_score=batch_stereotypes if batch_stereotypes else {}  
    )
    db.add(session)
    db.commit()

    return response

@app.patch("/update_viewing_time")
async def update_viewing_time(time_data: TimeUpdate, db: Session = Depends(get_db)):
    session = db.query(AnalysisSession).filter(
        AnalysisSession.website == time_data.website
    ).order_by(AnalysisSession.entry_time.desc()).first()  # Get latest session for the website

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if time_data.exit_time <= session.entry_time:
        raise HTTPException(status_code=400, detail="Exit time must be greater than entry time.")

    session.exit_time = time_data.exit_time
    session.viewing_time = round(session.exit_time - session.entry_time, 4)  # Ensure correct viewing time
    db.commit()

    return {"message": "Viewing time updated successfully"}

@app.get("/overall_results")
async def get_overall_accumulated_results(website: str = None, db: Session = Depends(get_db)):
    query = db.query(AnalysisSession).filter(AnalysisSession.viewing_time > 0)
    if website:
        query = query.filter(AnalysisSession.website == website)  # Filter by website if provided

    sessions = query.all()

    if not sessions:
        raise HTTPException(status_code=404, detail="No data found")

    total_time = sum(s.viewing_time for s in sessions)
    if total_time == 0:
        return {"message": "Total viewing time is zero, no meaningful scores available."}

    weighted_scores = {
        "sentiment": 0,
        "emotions": {},
        "political_bias": {},
        "stereotype": {}  
    }

    for session in sessions:
        weight = session.viewing_time / total_time
        weighted_scores["sentiment"] += session.sentiment_score * weight

        for category, target_key in [("emotions", "emotions"), ("political_bias", "political_bias"), ("stereotype_score", "stereotype")]:
            session_data = getattr(session, category)
            if not session_data:
                continue

            for key, score in session_data.items():
                if key not in weighted_scores[target_key]:
                    weighted_scores[target_key][key] = 0
                weighted_scores[target_key][key] += score * weight


    weighted_scores["sentiment"] = round(weighted_scores["sentiment"], 4)
    weighted_scores["emotions"] = {k: round(v, 4) for k, v in weighted_scores["emotions"].items()}
    weighted_scores["political_bias"] = {k: round(v, 4) for k, v in weighted_scores["political_bias"].items()}
    weighted_scores["stereotype"] = {k: round(v, 4) for k, v in weighted_scores["stereotype"].items()}  # ✅ Fixed access to correct key

    return {
        "website": website if website else "All websites",
        "total_viewing_time": round(total_time, 4),
        "weighted_scores": weighted_scores
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
