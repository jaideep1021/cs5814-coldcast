"""CPU prototype. Synthetic mode tests software, never establishes benchmark performance.

Real input: CSV with benchmark_id,item_id,subject_id,item_text,label,provider.
Use only permitted public releases. A normalized export must preserve repeated runs.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, re, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['benchmark_id', 'item_id', 'subject_id', 'item_text', 'label', 'provider']

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

def synthetic(seed=5814):
    rng = np.random.default_rng(seed)
    rows = []
    domains = ['algebra', 'logic', 'science', 'history']
    for b in range(8):
        for i in range(80):
            d, level = i % 4, int(rng.integers(1, 5))
            text = f'{domains[d]} task level {level} benchmark {b} unique {i} '
            text += ' '.join(['reasoning'] * level)
            difficulty = (level - 2.5) * .6 + rng.normal(0, .3) + .1 * (b - 3.5)
            for s in range(8):
                ability = (s - 3.5) * .35 + (.4 if s % 4 == d else 0)
                rows.append(dict(benchmark_id=f'b{b}', item_id=f'b{b}-i{i}',
                    subject_id=f's{s}', provider=f'p{s//2}', item_text=text,
                    label=int(rng.random() < sigmoid(ability - difficulty))))
    return rows

def validate(rows):
    if not rows:
        raise ValueError('Empty dataset')
    item_info, text_bench, subject_provider = {}, {}, {}
    for r in rows:
        if any(k not in r or str(r[k]).strip() == '' for k in FIELDS):
            raise ValueError('All required fields must be present and nonempty')
        if str(r['label']) not in ('0', '1'):
            raise ValueError('Labels must be binary 0/1')
        r['label'] = int(r['label'])
        key = (r['benchmark_id'], r['item_id'])
        text = ' '.join(r['item_text'].lower().split())
        if key in item_info and item_info[key] != text:
            raise ValueError('Conflicting text for an item ID')
        item_info[key] = text
        if text in text_bench and text_bench[text] != r['benchmark_id']:
            raise ValueError('Duplicate normalized text crosses benchmarks; group/remove before splitting')
        text_bench[text] = r['benchmark_id']
        if r['subject_id'] in subject_provider and subject_provider[r['subject_id']] != r['provider']:
            raise ValueError('Conflicting provider for subject')
        subject_provider[r['subject_id']] = r['provider']
    return rows

def split(rows, seed):
    groups = np.array(sorted({r['benchmark_id'] for r in rows}))
    if len(groups) < 5:
        raise ValueError('Need at least 5 benchmarks for train/development/calibration/test')
    np.random.default_rng(seed).shuffle(groups)
    n = max(1, len(groups) // 8)
    parts = dict(train=set(groups[3*n:]), development=set(groups[:n]),
                 calibration=set(groups[n:2*n]), test=set(groups[2*n:3*n]))
    result = {k: [r for r in rows if r['benchmark_id'] in ids] for k, ids in parts.items()}
    for a in parts:
        for b in parts:
            if a != b:
                assert parts[a].isdisjoint(parts[b])
    return result, {k: sorted(v) for k,v in parts.items()}

class Features:
    """Fixed text hashing + training-only category vocabularies; no item/benchmark IDs."""
    def __init__(self, train, dim=128):
        self.dim = dim
        self.vocab = {k: {v: i+1 for i,v in enumerate(sorted({r[k] for r in train}))}
                      for k in ('subject_id', 'provider')}
    def transform(self, rows):
        text = np.zeros((len(rows), self.dim))
        meta = np.zeros((len(rows), sum(len(v)+1 for v in self.vocab.values())))
        for i,r in enumerate(rows):
            for token in re.findall(r'\b\w+\b', r['item_text'].lower()):
                # Stable across processes; Python's randomized hash() is deliberately avoided.
                h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little')
                text[i,h % self.dim] += 1 if (h >> 8) % 2 else -1
            text[i] /= max(1, np.linalg.norm(text[i]))
            offset = 0
            for k,v in self.vocab.items():
                meta[i,offset + v.get(r[k], 0)] = 1
                offset += len(v)+1
        return text, meta

class Network:
    def __init__(self, sizes, seed):
        rng = np.random.default_rng(seed)
        self.w = [rng.normal(0, np.sqrt(2/a), (a,b)) for a,b in zip(sizes[:-1],sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
    def forward(self, x):
        activations = [x]
        for w,b in zip(self.w[:-1], self.b[:-1]):
            activations.append(np.tanh(activations[-1] @ w + b))
        logits = (activations[-1] @ self.w[-1] + self.b[-1]).ravel()
        return logits, activations
    def gradients(self, x, y, decay=1e-4):
        z,a = self.forward(x)
        delta = ((sigmoid(z) - y)/len(y))[:,None]
        dw,db = [None]*len(self.w), [None]*len(self.w)
        for i in reversed(range(len(self.w))):
            dw[i] = a[i].T @ delta + decay*self.w[i]
            db[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.w[i].T) * (1-a[i]**2)
        return dw,db
    def fit(self,x,y,xd,yd,epochs=120):
        params = self.w+self.b
        m,v = [np.zeros_like(p) for p in params],[np.zeros_like(p) for p in params]
        best, saved, best_epoch = np.inf,None,0
        for t in range(1,epochs+1):
            dw,db = self.gradients(x,y)
            for j,(p,g) in enumerate(zip(params,dw+db)):
                m[j] = .9*m[j]+.1*g
                v[j] = .999*v[j]+.001*g*g
                p -= .01*(m[j]/(1-.9**t))/(np.sqrt(v[j]/(1-.999**t))+1e-8)
            score = np.mean((sigmoid(self.forward(xd)[0])-yd)**2)
            if score < best:
                best,saved,best_epoch = score,[p.copy() for p in params],t
        for p,s in zip(params,saved):
            p[:] = s
        return best_epoch

def metrics(y,p):
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid probabilities')
    p = np.clip(p,1e-7,1-1e-7)
    bins = np.minimum((p*10).astype(int),9)
    reliability = []
    for b in range(10):
        m = bins == b
        if m.any():
            reliability.append(dict(bin=b, count=int(m.sum()), confidence=float(p[m].mean()), accuracy=float(y[m].mean())))
    return dict(brier=float(np.mean((p-y)**2)), log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                ece=float(sum(r['count']*abs(r['confidence']-r['accuracy']) for r in reliability)/len(y)),
                reliability=reliability)

def bootstrap_delta(rows,y,p,ref,seed):
    # All subjects and repeated responses of each item move together.
    ids = sorted({(r['benchmark_id'],r['item_id']) for r in rows})
    groups = [[i for i,r in enumerate(rows) if (r['benchmark_id'],r['item_id'])==key] for key in ids]
    delta = (p-y)**2-(ref-y)**2
    sums = np.array([delta[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    draws = np.random.default_rng(seed).integers(0,len(groups),(500,len(groups)))
    values = sums[draws].sum(1)/counts[draws].sum(1)
    return list(np.quantile(values,[.025,.975]).astype(float))

def run(rows,out,seed,provenance):
    start = time.time()
    validate(rows)
    parts,manifest = split(rows,seed)
    features = Features(parts['train'])
    xs = {k:features.transform(v) for k,v in parts.items()}
    ys = {k:np.array([r['label'] for r in v]) for k,v in parts.items()}
    y = ys['test']
    global_rate = float(ys['train'].mean())
    rates = {}
    for s in {r['subject_id'] for r in parts['train']}:
        labels = [r['label'] for r in parts['train'] if r['subject_id']==s]
        rates[s] = (sum(labels)+10*global_rate)/(len(labels)+10)
    preds = {'constant_0.5': np.full(len(y),.5), 'global_rate':np.full(len(y),global_rate),
             'system_rate':np.array([rates.get(r['subject_id'],global_rate) for r in parts['test']])}
    settings, models = {},{}
    for name,index,hidden in [('text_logistic',0,[]),('metadata_logistic',1,[]),('fusion_mlp',None,[64,32])]:
        x = {k:np.hstack(v) if index is None else v[index] for k,v in xs.items()}
        model = Network([x['train'].shape[1]]+hidden+[1],seed)
        epoch = model.fit(x['train'],ys['train'],x['development'],ys['development'])
        preds[name] = sigmoid(model.forward(x['test'])[0])
        settings[name] = dict(selected_epoch=epoch,hidden=hidden)
        models[name] = model
        if name=='fusion_mlp':
            zc = model.forward(x['calibration'])[0]
            grid = np.exp(np.linspace(np.log(.2),np.log(5),81))
            temp = float(min(grid,key=lambda t:metrics(ys['calibration'],sigmoid(zc/t))['log_loss']))
            preds['fusion_temperature'] = sigmoid(model.forward(x['test'])[0]/temp)
            settings['temperature'] = temp
    results = {k:metrics(y,p) for k,p in preds.items()}
    for k,p in preds.items():
        results[k]['brier_delta_vs_system_95ci'] = bootstrap_delta(parts['test'],y,p,preds['system_rate'],seed)
        pair_metrics=[]
        for pair in sorted({(r['subject_id'],r['benchmark_id']) for r in parts['test']}):
            m=np.array([(r['subject_id'],r['benchmark_id'])==pair for r in parts['test']])
            pair_metrics.append(float(np.mean((p[m]-y[m])**2)))
        results[k]['macro_pair_brier'] = float(np.mean(pair_metrics))
    out.mkdir(parents=True,exist_ok=True)
    for name,model in models.items():
        np.savez(out/(name+'.npz'),**{f'w{i}':w for i,w in enumerate(model.w)},**{f'b{i}':b for i,b in enumerate(model.b)})
    (out/'features.json').write_text(json.dumps(dict(text_hash_dimensions=features.dim,vocab=features.vocab),indent=2))
    fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    summary=dict(provenance=provenance,seed=seed,data_sha256=fingerprint,
        split=manifest,counts={k:dict(rows=len(v),items=len({(r['benchmark_id'],r['item_id']) for r in v}),
            subjects=len({r['subject_id'] for r in v}),positive_rate=float(ys[k].mean())) for k,v in parts.items()},
        settings=settings,metrics=results,elapsed_seconds=round(time.time()-start,3),
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['benchmark_id','item_id','subject_id','label']+list(preds))
        writer.writeheader()
        for i,r in enumerate(parts['test']):
            writer.writerow({**{k:r[k] for k in ['benchmark_id','item_id','subject_id','label']},**{k:float(p[i]) for k,p in preds.items()}})
    print(json.dumps({k:{m:round(v[m],5) for m in ['brier','log_loss','ece']} for k,v in results.items()},indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic',action='store_true')
    group.add_argument('--csv',type=Path)
    parser.add_argument('--seed',type=int,default=5814)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.synthetic:
        rows=synthetic(args.seed)
        provenance='SYNTHETIC SOFTWARE CHECK ONLY; not official competition data or scientific evidence'
    else:
        with args.csv.open(newline='') as f:
            rows=list(csv.DictReader(f))
        provenance=f'User-supplied normalized CSV: {args.csv.name}; eligibility must be verified separately'
    out=args.out or ROOT/'results'/('synthetic' if args.synthetic else 'real')
    run(rows,out,args.seed,provenance)
