from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from recommender import get_recommendations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(query: str = Query(...)):
    return get_recommendations(query)