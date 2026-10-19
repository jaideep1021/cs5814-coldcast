"""Real-data pilot: fixed grouped splits, baselines, PyTorch fusion, retrieval, calibration.

No experiment setting is selected with test labels. All released model metrics include
negative findings. The primary five-fold experiment holds out entire source benchmarks;
the secondary experiment holds out exact-normalized-text groups within each source.
"""
from pathlib import Path
import copy,json,time,hashlib,platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
import joblib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/real'
SEEDS=[5814,5815,5816]
CATS=['normalized_name','provider','harness','reasoning_effort','harness_version','system_key']
EPOCHS=40

def measures(y,p,groups):
    p=np.asarray(p);y=np.asarray(y)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    sq=(y-p)**2
    _,g=np.unique(groups,return_inverse=True)
    macro=float(np.mean(np.bincount(g,weights=sq)/np.bincount(g)))
    bins=np.minimum((p*10).astype(int),9);cal=[]
    for b in range(10):
        m=bins==b
        if m.any():cal.append(dict(bin=b,count=int(m.sum()),confidence=float(p[m].mean()),accuracy=float(y[m].mean())))
    q=np.clip(p,1e-7,1-1e-7)
    return dict(brier=float(sq.mean()),macro_pair_brier=macro,
        log_loss=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),
        ece=sum(v['count']*abs(v['confidence']-v['accuracy']) for v in cal)/len(y),reliability=cal)

def partitions(df):
    sources=sorted(df.source.unique());out=[]
    for i,test in enumerate(sources):
        dev=sources[(i+1)%len(sources)];cal=sources[(i+2)%len(sources)]
        sets=dict(test=[test],development=[dev],calibration=[cal],train=[s for s in sources if s not in [test,dev,cal]])
        out.append(('benchmark',test,{k:np.flatnonzero(df.source.isin(v)) for k,v in sets.items()}))
    idx={k:[] for k in ['train','development','calibration','test']}
    for source,rows in df.groupby('source'):
        ids=np.array(sorted(rows.text_group.unique()));np.random.default_rng(5814).shuffle(ids)
        n=len(ids);n1=int(.6*n);n2=int(.75*n);n3=int(.85*n)
        for key,part in zip(idx,[ids[:n1],ids[n1:n2],ids[n2:n3],ids[n3:]]):
            idx[key].extend(rows.index[rows.text_group.isin(part)].tolist())
    out.append(('item','within_benchmark',{k:np.array(v,dtype=int) for k,v in idx.items()}))
    for protocol,name,parts in out:
        assert sum(map(len,parts.values()))==len(df)
        for a in parts:
            for b in parts:
                if a!=b:
                    assert set(df.iloc[parts[a]].text_group).isdisjoint(df.iloc[parts[b]].text_group)
                    if protocol=='benchmark': assert set(df.iloc[parts[a]].source).isdisjoint(df.iloc[parts[b]].source)
    return out

def category_features(df,train):
    vocab={c:{v:i+1 for i,v in enumerate(sorted(set(df.iloc[train][c])))} for c in CATS}
    codes=np.array([[vocab[c].get(row[c],0) for c in CATS] for row in df[CATS].to_dict('records')],dtype=np.int64)
    return vocab,codes

def retrieval(df,emb,train):
    """Training-only neighbor correctness; leave same text group out for training queries."""
    table=df.iloc[train].groupby('text_group').agg(rate=('label','mean')).reset_index()
    rowmap=df.drop_duplicates('text_group').set_index('text_group').index
    first=df.reset_index().drop_duplicates('text_group').set_index('text_group')['index']
    bank=emb[first.loc[table.text_group].to_numpy()]
    unique=df.drop_duplicates('text_group')
    values={};global_rate=df.iloc[train].label.mean()
    for start in range(0,len(unique),128):
        chunk=unique.iloc[start:start+128];sim=emb[chunk.index]@bank.T
        for j,key in enumerate(chunk.text_group):
            sim[j,table.text_group.to_numpy()==key]=-np.inf
            k=min(10,len(table)-1);near=np.argpartition(sim[j],-k)[-k:]
            w=np.exp((sim[j,near]-np.max(sim[j,near]))/.15)
            values[key]=float((w@table.rate.to_numpy()[near]+global_rate)/(w.sum()+1))
    return np.array([values[k] for k in df.text_group],dtype=np.float32)

class Fusion(nn.Module):
    def __init__(self,vocab,with_retrieval=False):
        super().__init__();self.with_retrieval=with_retrieval
        self.emb=nn.ModuleList([nn.Embedding(len(vocab[c])+1,8) for c in CATS])
        d=384+8*len(CATS)+int(with_retrieval)
        self.head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x,c,r):
        if self.training:
            c=c.clone();c[torch.rand(c.shape)<.1]=0
        inputs=[x]+[layer(c[:,j]) for j,layer in enumerate(self.emb)]
        if self.with_retrieval: inputs.append(r[:,None])
        return self.head(torch.cat(inputs,dim=1)).flatten()

def train_network(df,emb,codes,ret,parts,vocab,seed,use_ret):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=Fusion(vocab,use_ret);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
    x=torch.from_numpy(emb);c=torch.from_numpy(codes);r=torch.from_numpy(ret);y=torch.tensor(df.label.to_numpy(),dtype=torch.float32)
    best=np.inf;beststate=None;bestepoch=0;history=[]
    for epoch in range(1,EPOCHS+1):
        net.train();shuffled=rng.permutation(parts['train'])
        for a in range(0,len(shuffled),1024):
            ids=shuffled[a:a+1024];opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(net(x[ids],c[ids],r[ids]),y[ids])
            loss.backward();opt.step()
        net.eval()
        with torch.inference_mode():
            ids=parts['development'];p=torch.sigmoid(net(x[ids],c[ids],r[ids])).numpy()
        group=(df.iloc[ids].source+'|'+df.iloc[ids].system_key).to_numpy()
        score=measures(df.iloc[ids].label,p,group)['macro_pair_brier'];history.append(score)
        if score<best-1e-6:best=score;bestepoch=epoch;beststate=copy.deepcopy(net.state_dict())
        if epoch-bestepoch>=7:break
    net.load_state_dict(beststate);net.eval();z=[]
    with torch.inference_mode():
        for a in range(0,len(df),4096):z.append(net(x[a:a+4096],c[a:a+4096],r[a:a+4096]).numpy())
    return net,np.concatenate(z),dict(selected_epoch=bestepoch,epochs_run=epoch,development_macro_brier=best,history=history)

def temperature(z,y):
    grid=np.exp(np.linspace(np.log(.2),np.log(5),81))
    return float(min(grid,key=lambda t:np.mean(np.logaddexp(0,z/t)-y*z/t)))

def main():
    started=time.time();torch.set_num_threads(4)
    df=pd.read_parquet(ROOT/'data/real_pilot.parquet').reset_index(drop=True)
    cache=np.load(ROOT/'data/real_embeddings.npz');lookup={k:i for i,k in enumerate(cache['keys'])}
    emb=cache['vectors'][[lookup[k] for k in df.text_group]].astype(np.float32)
    OUT.mkdir(exist_ok=True);modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
    rows=[];allpred=[];manifests=[]
    for protocol,fold,parts in partitions(df):
        print(f'Running {protocol}/{fold}: '+str({k:len(v) for k,v in parts.items()}),flush=True)
        directory=modeldir/f'{protocol}_{fold}';directory.mkdir(exist_ok=True)
        tr=parts['train'];te=parts['test'];cal=parts['calibration']
        y=df.label.to_numpy();vocab,codes=category_features(df,tr);ret=retrieval(df,emb,tr)
        rate=float(y[tr].mean());training=df.iloc[tr]
        sys=training.groupby('system_key').label.agg(['sum','count'])
        rates=((sys['sum']+10*rate)/(sys['count']+10)).to_dict()
        preds={'constant_0.5':np.full(len(df),.5),'global_rate':np.full(len(df),rate),
            'system_rate':df.system_key.map(rates).fillna(rate).to_numpy(),'retrieval':ret}
        onehot=OneHotEncoder(handle_unknown='ignore');meta=onehot.fit_transform(df.iloc[tr][CATS])
        clf=LogisticRegression(C=1,max_iter=500).fit(meta,y[tr])
        preds['metadata_logistic']=clf.predict_proba(onehot.transform(df[CATS]))[:,1]
        joblib.dump((onehot,clf),directory/'metadata.joblib')
        semantic=LogisticRegression(C=1,max_iter=500).fit(emb[tr],y[tr])
        preds['semantic_logistic']=semantic.predict_proba(emb)[:,1];joblib.dump(semantic,directory/'semantic.joblib')
        vec=TfidfVectorizer(max_features=4096,min_df=2,ngram_range=(1,1),sublinear_tf=True)
        # Fit IDF to unique TRAINING items, not held-out data or repeated responses.
        vec.fit(training.drop_duplicates('text_group').item_text)
        tfidf=vec.transform(df.item_text)
        lex=LogisticRegression(C=1,max_iter=500).fit(tfidf[tr],y[tr])
        preds['tfidf_logistic']=lex.predict_proba(tfidf)[:,1];joblib.dump((vec,lex),directory/'tfidf.joblib')
        manifest=dict(protocol=protocol,fold=fold,
            partitions={k:dict(rows=len(ids),sources=sorted(df.iloc[ids].source.unique()),
                text_groups=sorted(df.iloc[ids].text_group.unique())) for k,ids in parts.items()},
            unseen_system_response_fraction=float((~df.iloc[te].system_key.isin(training.system_key)).mean()),
            unseen_model_response_fraction=float((~df.iloc[te].normalized_name.isin(training.normalized_name)).mean()))
        manifests.append(manifest)
        (directory/'vocab.json').write_text(json.dumps(vocab,indent=2))
        group=(df.iloc[te].source+'|'+df.iloc[te].system_key).to_numpy()
        for seed in SEEDS:
            current=preds.copy();settings={}
            for name,with_ret in [('fusion',False),('fusion_retrieval',True)]:
                model,z,details=train_network(df,emb,codes,ret,parts,vocab,seed,with_ret)
                t=temperature(z[cal],y[cal]);details['temperature']=t
                current[name]=expit(z);current[name+'_temperature']=expit(z/t)
                torch.save(dict(state_dict=model.state_dict(),vocab=vocab,with_retrieval=with_ret,temperature=t),directory/f'{name}_{seed}.pt')
                settings[name]=details
            (directory/f'training_{seed}.json').write_text(json.dumps(settings,indent=2))
            predframe=df.iloc[te][['source','item_id','text_group','response_id','system_key','label']].copy()
            predframe['protocol']=protocol;predframe['fold']=fold;predframe['seed']=seed
            for name,p in current.items():
                m=measures(y[te],p[te],group)
                rows.append(dict(protocol=protocol,fold=fold,seed=seed,model=name,**m))
                predframe[name]=p[te]
            allpred.append(predframe)
            print(f'  seed {seed}: fusion={rows[-4]["macro_pair_brier"]:.4f}; fusion+retrieval+T={rows[-1]["macro_pair_brier"]:.4f}',flush=True)
    pd.concat(allpred,ignore_index=True).to_parquet(OUT/'test_predictions.parquet',index=False)
    (OUT/'metrics_all.json').write_text(json.dumps(rows,indent=2))
    (OUT/'split_manifest.json').write_text(json.dumps(manifests,indent=2))
    summary=pd.DataFrame([{k:v for k,v in r.items() if k!='reliability'} for r in rows])
    summary.to_csv(OUT/'metrics_all.csv',index=False)
    # Report equal weight across benchmark folds, then mean/std across training seeds.
    seedmeans=summary.groupby(['protocol','model','seed'])[['macro_pair_brier','brier','log_loss','ece']].mean().reset_index()
    agg=seedmeans.groupby(['protocol','model']).agg(macro_brier_mean=('macro_pair_brier','mean'),
        macro_brier_seed_sd=('macro_pair_brier','std'),response_brier_mean=('brier','mean'),log_loss_mean=('log_loss','mean'),ece_mean=('ece','mean')).reset_index()
    agg.to_csv(OUT/'summary.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(seeds=SEEDS,max_epochs=EPOCHS,early_stopping_patience=7,
        learning_rate=.001,batch_size=1024,hidden=[64,32],metadata_embedding_dimensions=8,category_dropout=.1,
        dropout=.15,weight_decay=1e-4,selection='development macro subject-benchmark Brier',
        calibration='calibration response-weighted log loss',retrieval_k=10,retrieval_temperature=.15,
        torch=torch.__version__,python=platform.python_version(),elapsed_seconds=time.time()-started),indent=2))
    print(agg.to_string(index=False),flush=True)

if __name__=='__main__':main()
