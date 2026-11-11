import os
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI,HTTPException,Query
from hookrec.data import load_dataset
from hookrec.recommender import HybridArtistRecommender
DATA_DIR=Path(os.getenv('HOOKREC_DATA_DIR','data/raw/lastfm-2k'))

def _to_builtin(value):
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_builtin(value.tolist())
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value

app=FastAPI(title='HookRec API',description='Hybrid music-artist recommendation MVP.',version='0.1.0')
@lru_cache(maxsize=1)
def model():
    interactions,artists,tags=load_dataset(DATA_DIR)
    return HybridArtistRecommender(interactions,artists,tags)
@app.get('/health')
def health():
    try: m=model()
    except FileNotFoundError as e: raise HTTPException(503,str(e)) from e
    return {'status':'ok','dimensions':m.dimensions}
@app.get('/recommend/{user_id}')
def recommend(user_id:int,k:int=Query(10,ge=1,le=100),strategy:str=Query('hybrid',pattern='^(hybrid|collaborative|content)$')):
    try: results=model().recommend(user_id,k,strategy)
    except FileNotFoundError as e: raise HTTPException(503,str(e)) from e
    return _to_builtin({'user_id':user_id,'strategy':strategy,'recommendations':[r.__dict__ for r in results]})
@app.get('/similar/{artist_id}')
def similar(artist_id:int,k:int=Query(10,ge=1,le=100)):
    try: results=model().similar_artists(artist_id,k)
    except FileNotFoundError as e: raise HTTPException(503,str(e)) from e
    except KeyError as e: raise HTTPException(404,str(e)) from e
    return _to_builtin({'artist_id':artist_id,'similar_artists':[r.__dict__ for r in results]})
