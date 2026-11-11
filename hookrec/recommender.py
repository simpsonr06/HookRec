from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

@dataclass
class Recommendation:
    artist_id: int
    artist_name: str
    score: float

class HybridArtistRecommender:
    """Implicit item-item CF + TF-IDF tag similarity hybrid recommender."""
    def __init__(self, interactions: pd.DataFrame, artists: pd.DataFrame,
                 tag_rows: pd.DataFrame | None = None, content_weight: float = .35):
        if not 0 <= content_weight <= 1:
            raise ValueError('content_weight must be between 0 and 1')
        if not {'user_id','artist_id','weight'} <= set(interactions.columns):
            raise ValueError('interactions need user_id, artist_id, weight')
        if not {'artist_id','artist_name'} <= set(artists.columns):
            raise ValueError('artists need artist_id, artist_name')
        self.content_weight = content_weight
        self.artist_ids = sorted(artists.artist_id.astype(int).unique())
        self.user_ids = sorted(interactions.user_id.astype(int).unique())
        self.artist_index = {v:i for i,v in enumerate(self.artist_ids)}
        self.user_index = {v:i for i,v in enumerate(self.user_ids)}
        self.artist_names = dict(zip(artists.artist_id.astype(int), artists.artist_name.astype(str)))
        known = interactions[interactions.artist_id.isin(self.artist_index)].copy()
        known['strength'] = np.log1p(known.weight.clip(lower=0))
        rows = known.user_id.map(self.user_index).to_numpy()
        cols = known.artist_id.map(self.artist_index).to_numpy()
        self.user_item = csr_matrix((known.strength.to_numpy(float),(rows,cols)),
                                    shape=(len(self.user_ids),len(self.artist_ids)))
        item_user = normalize(self.user_item.T.tocsr(), norm='l2', axis=1)
        self.cf_similarity = (item_user @ item_user.T).tocsr()
        self.cf_similarity.setdiag(0); self.cf_similarity.eliminate_zeros()
        self.popularity = np.asarray(self.user_item.sum(axis=0)).ravel()
        self.popularity = self._norm(self.popularity)
        self.content_similarity = self._content(tag_rows)

    @staticmethod
    def _norm(x):
        x = np.asarray(x, dtype=float)
        m = float(x.max()) if x.size else 0.0
        return x / m if m > 0 else np.zeros_like(x)

    def _content(self, tags):
        n = len(self.artist_ids)
        if tags is None or tags.empty:
            return csr_matrix((n,n), dtype=float)
        docs = tags.groupby('artist_id').tag.apply(lambda s: ' '.join(map(str,s)))
        corpus = [docs.get(a, '') for a in self.artist_ids]
        if not any(x.strip() for x in corpus):
            return csr_matrix((n,n), dtype=float)
        tfidf = TfidfVectorizer(lowercase=True, stop_words='english', min_df=1,
                                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9&'-]+\b")
        matrix = normalize(tfidf.fit_transform(corpus))
        sim = (matrix @ matrix.T).tocsr()
        sim.setdiag(0); sim.eliminate_zeros()
        return sim

    def _profile(self, user_id):
        idx = self.user_index.get(int(user_id))
        return None if idx is None else self.user_item.getrow(idx)

    def recommend(self, user_id: int, k: int=10, strategy: str='hybrid'):
        if k < 1: raise ValueError('k must be >= 1')
        if strategy not in {'hybrid','collaborative','content'}: raise ValueError('unknown strategy')
        profile = self._profile(user_id)
        if profile is None or profile.nnz == 0:
            scores = self.popularity.copy()
        else:
            cf = self._norm((profile @ self.cf_similarity).toarray().ravel())
            cb = self._norm((profile @ self.content_similarity).toarray().ravel())
            if strategy == 'collaborative': scores = cf
            elif strategy == 'content': scores = cb
            else: scores = (1-self.content_weight)*cf + self.content_weight*cb
            if not np.any(scores): scores = self.popularity.copy()
            scores[profile.indices] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        order = candidates[np.argsort(scores[candidates])[::-1][:k]]
        return [Recommendation(self.artist_ids[i], self.artist_names.get(self.artist_ids[i], 'Unknown'), float(scores[i])) for i in order]

    def similar_artists(self, artist_id: int, k: int=10):
        if artist_id not in self.artist_index: raise KeyError(f'Unknown artist_id: {artist_id}')
        if k < 1: raise ValueError('k must be >= 1')
        i = self.artist_index[artist_id]
        scores = self.content_similarity.getrow(i).toarray().ravel()
        order = np.argsort(scores)[::-1]
        out=[]
        for j in order:
            if j == i or scores[j] <= 0: continue
            aid=self.artist_ids[j]
            out.append(Recommendation(aid,self.artist_names.get(aid,'Unknown'),float(scores[j])))
            if len(out) >= k: break
        return out

    @property
    def dimensions(self):
        return {'users':len(self.user_ids),'artists':len(self.artist_ids),'interactions':int(self.user_item.nnz)}
