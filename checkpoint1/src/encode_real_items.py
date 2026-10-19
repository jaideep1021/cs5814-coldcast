"""Pinned local MiniLM encoding, with explicit head/tail truncation for long prompts."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download,hf_hub_url,get_hf_file_metadata
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parents[1]
MODEL='sentence-transformers/all-MiniLM-L6-v2'

def main():
    start=time.time();torch.set_num_threads(4)
    dest=ROOT/'data/minilm'
    saved=ROOT/'results/real/encoder.json'
    revision=(json.loads(saved.read_text())['revision'] if saved.exists() else
        get_hf_file_metadata(hf_hub_url(MODEL,'config.json'),timeout=20).commit_hash)
    for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']:
        hf_hub_download(MODEL,name,revision=revision,local_dir=dest)
        print('Model asset ready:',name,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(dest,local_files_only=True)
    model=AutoModel.from_pretrained(dest,local_files_only=True).eval()
    items=pd.read_parquet(ROOT/'data/real_items.parquet')
    sequences=[];lengths=[]
    for text in items.item_text:
        ids=tokenizer.encode(text,add_special_tokens=False,truncation=False)
        lengths.append(len(ids))
        if len(ids)>254:ids=ids[:127]+ids[-127:]
        sequences.append({'input_ids':[tokenizer.cls_token_id]+ids+[tokenizer.sep_token_id]})
    vectors=[]
    with torch.inference_mode():
        for startidx in range(0,len(sequences),32):
            batch=tokenizer.pad(sequences[startidx:startidx+32],padding=True,return_tensors='pt')
            hidden=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled,p=2,dim=1).numpy())
            if startidx%320==0: print(f'Encoded {min(startidx+32,len(items))}/{len(items)} items',flush=True)
    np.savez(ROOT/'data/real_embeddings.npz',keys=items.text_group.to_numpy(dtype=str),vectors=np.concatenate(vectors))
    info=dict(model=MODEL,revision=revision,dimensions=384,items=len(items),max_tokens=256,
        truncation='If >254 tokens, retain first 127 and last 127, plus CLS/SEP; masked mean pooling and L2 normalization',
        truncated_items=int(sum(n>254 for n in lengths)),token_length_median=float(np.median(lengths)),
        token_length_max=int(max(lengths)),seconds=round(time.time()-start,2),device='cpu',fine_tuned=False)
    (ROOT/'results/real/encoder.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
