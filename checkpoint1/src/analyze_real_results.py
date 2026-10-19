"""Generate comparisons, plots, paired uncertainty and a random-label adaptation pilot."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_real_models import ROOT,OUT,measures

BUDGETS=[0,1,3,7,15,31]
WEIGHTS=np.array([.1,.2,.2,.2,.2,.1])
MODELS=['constant_0.5','system_rate','metadata_logistic','fusion','fusion_temperature','fusion_retrieval_temperature']

def acquisition_split(frame):
    keys=np.array(sorted(frame.text_group.unique()))
    identity=str(frame.source.iloc[0])+'|'+str(frame.system_key.iloc[0])
    seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed);rng.shuffle(keys)
    mid=len(keys)//2
    acquisition=keys[:mid];evaluation=set(keys[mid:])
    assert set(acquisition).isdisjoint(evaluation)
    # One randomly chosen recorded outcome per acquired distinct-text item.
    selected=[]
    for key in acquisition:
        rows=frame[frame.text_group==key]
        selected.append(rows.iloc[int(rng.integers(len(rows)))])
    return pd.DataFrame(selected),frame[frame.text_group.isin(evaluation)]

def adapt(base_eval,base_acq,labels):
    if not len(labels):return np.asarray(base_eval)
    z=logit(np.clip(base_acq,1e-6,1-1e-6));y=np.asarray(labels)
    # Fixed N(0,1) prior on a scalar ability correction; no test-label tuning.
    objective=lambda delta:np.sum(np.logaddexp(0,z+delta)-y*(z+delta))+.5*delta**2
    delta=minimize_scalar(objective,bounds=(-5,5),method='bounded').x
    return expit(logit(np.clip(base_eval,1e-6,1-1e-6))+delta)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plots-only',action='store_true');args=parser.parse_args()
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    df=pd.read_parquet(OUT/'test_predictions.parquet')
    summary=pd.read_csv(OUT/'summary.csv')
    base=pd.read_csv(OUT/'metrics_all.csv')
    plt.rcParams.update({'font.size':9,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,protocol,title in zip(axes,['item','benchmark'],['Held-out questions (secondary)','Held-out benchmarks (primary)']):
        r=summary[summary.protocol==protocol].sort_values('macro_brier_mean')
        ax.barh(r.model,r.macro_brier_mean,xerr=r.macro_brier_seed_sd,color='#287d8e')
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Macro Brier score (lower is better)');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(figs/'model_comparison.png');plt.close(fig)
    # Per-source mean scores identify the shift hidden by an overall average.
    cross=base[base.protocol=='benchmark'].groupby(['fold','model']).macro_pair_brier.mean().unstack()
    cross.to_csv(OUT/'per_benchmark.csv')
    fig,ax=plt.subplots(figsize=(9,4))
    cross[['constant_0.5','system_rate','fusion_temperature','fusion_retrieval_temperature']].plot.bar(ax=ax)
    ax.set_ylabel('Macro Brier');ax.set_title('Unseen-benchmark performance');ax.tick_params(axis='x',rotation=20)
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=7,title=None)
    fig.tight_layout();fig.savefig(figs/'benchmark_breakdown.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,protocol in zip(axes,['item','benchmark']):
        rows=df[(df.protocol==protocol)&(df.seed==5814)]
        ax.plot([0,1],[0,1],'--',color='grey')
        for model in ['system_rate','fusion','fusion_temperature']:
            stats=measures(rows.label,rows[model],rows.source+'|'+rows.system_key)['reliability']
            ax.plot([v['confidence'] for v in stats],[v['accuracy'] for v in stats],'o-',label=model)
        ax.set(title=protocol+' split; seed 5814',xlabel='Mean prediction',ylabel='Observed correctness',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figs/'reliability.png');plt.close(fig)
    # Paired equal-benchmark resampling of mean per-fold deltas; only five clusters.
    rng=np.random.default_rng(5814);ci=[]
    for model in cross.columns:
        delta=(cross[model]-cross.system_rate).to_numpy()
        sample=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
        low,high=np.quantile(sample,[.025,.975])
        ci.append(dict(model=model,mean_delta=float(delta.mean()),lower=float(low),upper=float(high),
            warning='Only five benchmark clusters; descriptive, unstable interval. Seeds averaged before resampling.'))
    (OUT/'paired_benchmark_intervals.json').write_text(json.dumps(ci,indent=2))
    if args.plots_only:return
    adaptation=[];pool_manifest=[]
    for (seed,source,system),frame in df[df.protocol=='benchmark'].groupby(['seed','source','system_key']):
        if frame.text_group.nunique()<80:continue
        acq,evaluation=acquisition_split(frame)
        assert len(acq)>=31
        if seed==5814:pool_manifest.append(dict(source=source,system_key=system,
            acquisition_groups=acq.text_group.tolist(),evaluation_groups=sorted(evaluation.text_group.unique())))
        for model in MODELS:
            values=[]
            for n in BUDGETS:
                p=adapt(evaluation[model],acq[model].iloc[:n],acq.label.iloc[:n])
                m=measures(evaluation.label,p,np.zeros(len(p)))
                values.append(m['brier'])
                adaptation.append(dict(seed=seed,source=source,system_key=system,model=model,budget=n,
                    brier=m['brier'],evaluation_rows=len(evaluation),evaluation_groups=evaluation.text_group.nunique()))
    ada=pd.DataFrame(adaptation);ada.to_csv(OUT/'adaptation_by_pair.csv',index=False)
    pools=json.dumps(pool_manifest,indent=2);(OUT/'adaptation_pools.json').write_text(pools)
    means=ada.groupby(['model','seed','budget']).brier.mean().reset_index()
    means.to_csv(OUT/'adaptation_curve.csv',index=False)
    alc=[]
    for (model,seed),group in means.groupby(['model','seed']):
        scores=group.set_index('budget').loc[BUDGETS].brier.to_numpy()
        alc.append(dict(model=model,seed=int(seed),brier_alc=float(WEIGHTS@scores),brier_zero=float(scores[0]),brier_31=float(scores[-1])))
    pd.DataFrame(alc).to_csv(OUT/'adaptation_alc.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    for model,rows in means.groupby('model'):
        v=rows.groupby('budget').brier.mean().reindex(BUDGETS)
        ax.plot(range(6),v,'o-',label=model)
    ax.set_xticks(range(6),BUDGETS);ax.set(xlabel='Revealed labels (log-spaced budgets)',ylabel='Pair-macro Brier',title='Local random-acquisition adaptation pilot')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figs/'adaptation.png');plt.close(fig)
    details=dict(eligible_pairs=len(pool_manifest),source_pair_counts=pd.Series([r['source'] for r in pool_manifest]).value_counts().to_dict(),
        budgets=BUDGETS,prior='Fixed unit Gaussian prior on scalar logit correction',acquisition='Random; one recorded response per distinct normalized-text item',
        protocol='Local offline analogue; no official competition score. Same evaluation pool at all budgets. Exact text groups replace raw IDs conservatively.')
    (OUT/'adaptation_protocol.json').write_text(json.dumps(details,indent=2));print(json.dumps(details,indent=2))
    print(pd.DataFrame(alc).groupby('model')[['brier_alc','brier_zero','brier_31']].mean().to_string())

if __name__=='__main__':main()
