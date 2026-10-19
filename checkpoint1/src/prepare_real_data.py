"""Audit pinned tables and build a label-independent, text-only binary pilot."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/measurement-db'
OUT=ROOT/'results/real'
SOURCES=['matharena','multi_swebench','real_webagents','researchcodebench','swe_rebench']
SEED=5814
MAX_ITEMS=300

def normalized_hash(text):
    return hashlib.sha256(' '.join(str(text).lower().split()).encode()).hexdigest()

def binary_math(criterion):
    try:
        values=json.loads(criterion)['response_scale']['values']
        return set(values)=={0,1}
    except (KeyError,TypeError,ValueError):return False

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit={'selection_seed':SEED,'maximum_item_text_groups_per_source':MAX_ITEMS,
        'excluded_collections':{'mmdocrag':'Fractional multidimensional judge scores, not binary correctness'},
        'sources':{},'predictor_fields':['item_text','normalized_name','provider','harness','reasoning_effort','harness_version','subject_id'],
        'excluded_predictor_fields':['grading_criterion','verifier','reference_answer','trace','response','item_features','interactors'],
        'selection_rule':'Small text-only pilot: uniformly sample up to 300 normalized-text groups per source before training; preserve all rows for selected groups.'}
    frames=[]
    for source in SOURCES:
        path=DATA/source/'formatted_tables'
        items=pd.read_parquet(path/'items.parquet')
        subjects=pd.read_parquet(path/'subjects.parquet')
        responses=pd.read_parquet(path/'responses.parquet')
        meta=yaml.safe_load((DATA/source/'metadata.yaml').read_text())['benchmark']
        assert meta['release']=='public'
        assert not items.item_id.duplicated().any()
        assert not subjects.subject_id.duplicated().any()
        assert not responses.response_id.duplicated().any()
        stats={'raw_items':len(items),'raw_responses':len(responses),'raw_subjects':len(subjects),
            'license':meta.get('license'),'release':meta['release'],'source_url':meta.get('data_source_url')}
        valid=items.content.notna() & items.content.fillna('').str.strip().ne('')
        stats['missing_text_items']=int((~valid).sum())
        if source=='matharena':
            mask=items.grading_criterion.map(binary_math)
            stats['nonbinary_rubric_items']=int((~mask).sum())
            valid &=mask
        # No images are fetched or encoded; exclude tasks with external assets.
        asset=items.asset_manifest.fillna('').str.strip().isin(['','[]','{}','null'])
        stats['items_with_assets']=int((~asset).sum())
        valid &=asset
        items=items.loc[valid].copy()
        items['text_group']=items.content.map(normalized_hash)
        validresp=responses[responses.item_id.isin(items.item_id)].copy()
        assert validresp.response.isin([0.,1.]).all(), f'Nonbinary outcomes in selected {source} items'
        assert validresp.subject_id.isin(subjects.subject_id).all()
        stats['eligible_items']=len(items); stats['eligible_responses']=len(validresp)
        joined=validresp.merge(items[['item_id','content','text_group']],on='item_id',validate='many_to_one')
        joined=joined.merge(subjects,on='subject_id',validate='many_to_one')
        joined['source']=source
        # Subject identity includes run condition; no response-derived features are used.
        joined['system_key']=[hashlib.sha256((str(s)+'|'+str(c or '')).encode()).hexdigest()[:20]
            for s,c in zip(joined.subject_id,joined.test_condition.fillna(''))]
        joined=joined.rename(columns={'content':'item_text','response':'label'})
        joined['label']=joined.label.astype(int)
        audit['sources'][source]=stats
        frames.append(joined)
    allrows=pd.concat(frames,ignore_index=True)
    # Conservative removal of text found in multiple source collections before sampling.
    cross=allrows.groupby('text_group').source.nunique()
    bad=set(cross[cross>1].index)
    audit['cross_source_duplicate_text_groups_removed']=len(bad)
    allrows=allrows[~allrows.text_group.isin(bad)].copy()
    selected=[]
    for source in SOURCES:
        rows=allrows[allrows.source==source]
        groups=np.array(sorted(rows.text_group.unique()))
        rng=np.random.default_rng(SEED)
        chosen=set(rng.choice(groups,min(MAX_ITEMS,len(groups)),replace=False))
        rows=rows[rows.text_group.isin(chosen)].copy()
        stats=audit['sources'][source]
        stats.update(selected_rows=len(rows),selected_items=int(rows.item_id.nunique()),
            selected_text_groups=int(rows.text_group.nunique()),selected_systems=int(rows.system_key.nunique()),
            positive_rate=float(rows.label.mean()),max_text_characters=int(rows.item_text.str.len().max()))
        selected.append(rows)
    result=pd.concat(selected,ignore_index=True).sort_values(['source','text_group','system_key','response_id']).reset_index(drop=True)
    keep=['source','benchmark_id','item_id','text_group','response_id','system_key','subject_id','item_text','label',
        'normalized_name','provider','harness','reasoning_effort','harness_version']
    result=result[keep].fillna('')
    result.to_parquet(ROOT/'data/real_pilot.parquet',index=False)
    # Deduplicated text table avoids repeating long paper text in the embedding cache.
    result[['text_group','item_text']].drop_duplicates('text_group').to_parquet(ROOT/'data/real_items.parquet',index=False)
    audit['total_rows']=len(result);audit['total_text_groups']=int(result.text_group.nunique())
    audit['data_revision']=json.loads((DATA/'coldcast_download_manifest.json').read_text())['revision']
    audit['pilot_sha256']=hashlib.sha256((ROOT/'data/real_pilot.parquet').read_bytes()).hexdigest()
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
