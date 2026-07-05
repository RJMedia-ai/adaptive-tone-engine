from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "project": "Adaptive Tone Engine",
        "status": "Day 1"
    }
