import argparse,json
from pathlib import Path
from hookrec.data import load_dataset
from hookrec.evaluation import leave_one_out_split,evaluate_model
from hookrec.recommender import HybridArtistRecommender

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='data/raw/lastfm-2k'); p.add_argument('--k',type=int,default=10); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',default='reports/evaluation.json'); a=p.parse_args()
    interactions,artists,tags=load_dataset(a.data_dir); train,test=leave_one_out_split(interactions,a.seed)
    model=HybridArtistRecommender(train,artists,tags); result=evaluate_model(model,test,a.k)
    result['dataset']={'source':'GroupLens HetRec 2011 Last.fm 2K','raw_interactions':len(interactions),'train_interactions':len(train),'held_out_users':len(test),'seed':a.seed,'split':'random leave-one-out per eligible user'}
    result['model_dimensions']=model.dimensions
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2)); print(f'\nSaved: {out}')
if __name__=='__main__': main()
