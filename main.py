from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

URL = os.getenv('URL')
API_KEY = os.getenv('API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class getMovie(BaseModel):
    MovieName:str

model = pd.read_pickle(open('movieList.pkl', 'rb')) 

cv = CountVectorizer(max_features=5000,stop_words="english")
vectors = cv.fit_transform(model['tags']).toarray()
similar_dist = cosine_similarity(vectors)

id_to_title = dict(zip(model['movie_id'], model['title']))


@app.get("/movieinfo")
async def geMovieInfo(page: int = 1, per_page: int = 4000):
    start = (page - 1) * per_page
    end = start + per_page
    movie_data = model.iloc[start:end]
    movies_list = movie_data[['movie_id', 'title']].to_dict('records')
    return {
        "movies": movies_list,
        "total_movies": len(model)
    }

@app.post("/")
async def recomandSimilarMovies(MovieName: getMovie):
    movie_index = model[model['title']==MovieName.MovieName].index[0]
    distance = similar_dist[movie_index]
    movie_lst = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:7]
    result = []
    for i in movie_lst:
        data = model.iloc[i[0]]
        result.append({"movie_id": data['movie_id'].tolist(),"title":data['title']})
    return {"Similar_Movies": result}

