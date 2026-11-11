import pytest
from hookrec.recommender import HybridArtistRecommender
def test_excludes_consumed_artists(toy_data):
    i,a,t=toy_data; m=HybridArtistRecommender(i,a,t); consumed=set(i[i.user_id==1].artist_id); assert all(r.artist_id not in consumed for r in m.recommend(1,5))
def test_strategies_return_at_most_k(toy_data):
    i,a,t=toy_data; m=HybridArtistRecommender(i,a,t)
    for strategy in ('hybrid','collaborative','content'): assert len(m.recommend(1,2,strategy))<=2
def test_unknown_user_popularity_fallback(toy_data):
    i,a,t=toy_data; m=HybridArtistRecommender(i,a,t); assert len(m.recommend(999,3))==3
def test_unknown_artist_raises(toy_data):
    i,a,t=toy_data; m=HybridArtistRecommender(i,a,t)
    with pytest.raises(KeyError): m.similar_artists(999)
def test_invalid_k_raises(toy_data):
    i,a,t=toy_data; m=HybridArtistRecommender(i,a,t)
    with pytest.raises(ValueError): m.recommend(1,0)
def test_invalid_weight_raises(toy_data):
    i,a,t=toy_data
    with pytest.raises(ValueError): HybridArtistRecommender(i,a,t,content_weight=1.5)
