import json
from collections import defaultdict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union,List

app = FastAPI()

def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name).half()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

sentiment_model, sentiment_tokenizer = load_model("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_model, emotion_tokenizer = load_model("j-hartmann/emotion-english-distilroberta-base")
political_model, political_tokenizer = load_model("premsa/political-bias-prediction-allsides-BERT")
stereo_model, stereo_tokenizer = load_model("wu981526092/Sentence-Level-Stereotype-Detector")

emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
stereo_labels = ["unrelated", "stereotype_gender", "stereotype_race", "stereotype_profession", "stereotype_religion"]

class TextInput(BaseModel):
    texts: Union[str, List[str]]

def classify_texts(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return scores

@app.post("/SMBSanalyze")
def analyze_text(input_data: TextInput):
    texts = input_data.texts

    sentiment_scores = classify_texts(sentiment_model, sentiment_tokenizer, texts)
    overall_sentiment_score = (sentiment_scores[:, 1] - sentiment_scores[:, 0]).mean().item()

    emotion_scores = classify_texts(emotion_model, emotion_tokenizer, texts)
    emotion_totals = {label: emotion_scores[:, i].mean().item() for i, label in enumerate(emotion_labels)}

    political_scores = classify_texts(political_model, political_tokenizer, texts)
    avg_political_bias = {
        "left": political_scores[:, 0].mean().item(),
        "center": political_scores[:, 1].mean().item(),
        "right": political_scores[:, 2].mean().item(),
    }

    stereotype_scores = classify_texts(stereo_model, stereo_tokenizer, texts)
    stereotype_totals = {label: stereotype_scores[:, i].mean().item() for i, label in enumerate(stereo_labels)}
    top_stereotypes = dict(sorted(stereotype_totals.items(), key=lambda x: x[1], reverse=True)[:3])

    return {
        "overall_sentiment_score": overall_sentiment_score,
        "dominating_emotions": emotion_totals,
        "political_bias": avg_political_bias,
        "stereotype_analysis": top_stereotypes
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
