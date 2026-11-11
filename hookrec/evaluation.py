import numpy as np
import pandas as pd

def leave_one_out_split(interactions, seed=42):
    rng=np.random.default_rng(seed); parts=[]; test={}
    for uid,g in interactions.groupby('user_id',sort=True):
        g=g.drop_duplicates('artist_id')
        if len(g)>=2:
            idx=int(rng.integers(0,len(g))); test[int(uid)]={int(g.iloc[idx].artist_id)}
            parts.append(g.drop(g.index[idx]))
        else: parts.append(g)
    return (pd.concat(parts,ignore_index=True) if parts else interactions.iloc[:0].copy()),test

def precision_at_k(ranked,relevant,k):
    if k<=0: raise ValueError('k must be positive')
    return sum(x in relevant for x in ranked[:k])/k

def recall_at_k(ranked,relevant,k):
    if k<=0: raise ValueError('k must be positive')
    return sum(x in relevant for x in ranked[:k])/len(relevant) if relevant else 0.0

def ndcg_at_k(ranked,relevant,k):
    if k<=0: raise ValueError('k must be positive')
    dcg=sum((1/np.log2(i+2)) for i,x in enumerate(ranked[:k]) if x in relevant)
    ideal=sum(1/np.log2(i+2) for i in range(min(len(relevant),k)))
    return float(dcg/ideal) if ideal else 0.0

def evaluate_model(model,test,k=10):
    rows=[]
    for uid,relevant in test.items():
        for strategy in ('hybrid','collaborative','content'):
            ids=[r.artist_id for r in model.recommend(uid,k,strategy)]
            rows.append((strategy,precision_at_k(ids,relevant,k),recall_at_k(ids,relevant,k),ndcg_at_k(ids,relevant,k)))
        profile=model._profile(uid); consumed=set(profile.indices.tolist())
        ids=[model.artist_ids[i] for i in np.argsort(model.popularity)[::-1] if i not in consumed][:k]
        rows.append(('popularity',precision_at_k(ids,relevant,k),recall_at_k(ids,relevant,k),ndcg_at_k(ids,relevant,k)))
    if not rows: raise ValueError('No eligible test users')
    frame=pd.DataFrame(rows,columns=['strategy','precision','recall','ndcg'])
    summary=frame.groupby('strategy')[['precision','recall','ndcg']].mean()
    return {'k':k,'evaluated_users':len(test),'metrics':{s:{m:round(float(summary.loc[s,m]),6) for m in ('precision','recall','ndcg')} for s in summary.index}}
