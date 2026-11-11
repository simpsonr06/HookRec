from hookrec.evaluation import leave_one_out_split,precision_at_k,recall_at_k,ndcg_at_k
def test_metrics():
    ranked=[4,2,8,1]; relevant={2,4}; assert precision_at_k(ranked,relevant,2)==1; assert recall_at_k(ranked,relevant,2)==1; assert 0<ndcg_at_k(ranked,relevant,4)<=1
def test_split_holds_out_one_per_eligible_user(toy_data):
    interactions,_,_=toy_data; train,test=leave_one_out_split(interactions,7); assert set(test)=={1,2,3,4}
    for uid,held in test.items(): assert held.isdisjoint(set(train[train.user_id==uid].artist_id))
