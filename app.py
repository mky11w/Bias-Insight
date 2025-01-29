import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Union, List

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


sentiment_model, sentiment_tokenizer = load_model("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_model, emotion_tokenizer = load_model("j-hartmann/emotion-english-distilroberta-base")
political_model, political_tokenizer = load_model("premsa/political-bias-prediction-allsides-BERT")
stereo_model, stereo_tokenizer = load_model("wu981526092/Sentence-Level-Stereotype-Detector")


emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
stereo_labels = ["unrelated", "stereotype_gender", "stereotype_race", "stereotype_profession", "stereotype_religion"]
political_labels = ["left", "center", "right"]


accumulated_scores = {
    "sentiment": [],
    "emotions": {label: [] for label in emotion_labels},
    "political_bias": {label: [] for label in political_labels},
    "stereotype": {label: [] for label in stereo_labels},
    "batch_count": 0
}

class TextInput(BaseModel):
    texts: Union[str, List[str]]

async def classify_texts(model, tokenizer, texts):
    if isinstance(texts, str):
        texts = [texts]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return scores

@app.post("/SMBSanalyze")
async def analyze_text(input_data: TextInput, accumulated: bool = Query(False)):
    global accumulated_scores
    
    try:
        texts = input_data.texts
        if not texts or (isinstance(texts, list) and all(not t.strip() for t in texts)):
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")


        sentiment_scores = await classify_texts(sentiment_model, sentiment_tokenizer, texts)
        batch_sentiment_score = (sentiment_scores[:, 1] - sentiment_scores[:, 0]).mean().item()
        accumulated_scores["sentiment"].append(batch_sentiment_score)


        emotion_scores = await classify_texts(emotion_model, emotion_tokenizer, texts)
        batch_emotions = {label: round(emotion_scores[:, i].mean().item(), 4) for i, label in enumerate(emotion_labels)}
        for label, score in batch_emotions.items():
            accumulated_scores["emotions"][label].append(score)


        political_scores = await classify_texts(political_model, political_tokenizer, texts)
        batch_political_bias = {label: round(political_scores[:, i].mean().item(), 4) for i, label in enumerate(political_labels)}
        for label, score in batch_political_bias.items():
            accumulated_scores["political_bias"][label].append(score)


        stereotype_scores = await classify_texts(stereo_model, stereo_tokenizer, texts)
        batch_stereotypes = {label: round(stereotype_scores[:, i].mean().item(), 4) for i, label in enumerate(stereo_labels)}
        for label, score in batch_stereotypes.items():
            accumulated_scores["stereotype"][label].append(score)


        accumulated_scores["batch_count"] += 1


        response = {
            "batch_result": {
                "batch_sentiment_score": round(batch_sentiment_score, 4),
                "batch_dominating_emotions": batch_emotions,
                "batch_political_bias": batch_political_bias,
                "batch_stereotype_analysis": batch_stereotypes
            }
        }

        if accumulated:
            avg_sentiment_score = round(sum(accumulated_scores["sentiment"]) / accumulated_scores["batch_count"], 4)
            avg_emotions = {label: round(sum(scores) / accumulated_scores["batch_count"], 4) for label, scores in accumulated_scores["emotions"].items()}
            avg_political_bias = {label: round(sum(scores) / accumulated_scores["batch_count"], 4) for label, scores in accumulated_scores["political_bias"].items()}
            avg_stereotypes = {label: round(sum(scores) / accumulated_scores["batch_count"], 4) for label, scores in accumulated_scores["stereotype"].items()}
            
            response["accumulated_result"] = {
                "overall_sentiment_score": avg_sentiment_score,
                "dominating_emotions": avg_emotions,
                "political_bias": avg_political_bias,
                "stereotype_analysis": avg_stereotypes
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
