"""Offline local inference for a saved fusion checkpoint (not an organizer package).

Example: python src/predict_real.py --input data/example_input.json
Input JSON contains item_text and any known CATS fields. Missing/unseen categories
map to index zero. system_key is the prepared-data configuration key, if known.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from train_real_models import ROOT,CATS,Fusion
from scipy.special import expit

class Predictor:
    def __init__(self,checkpoint=None):
        torch.set_num_threads(4)
        checkpoint=checkpoint or ROOT/'results/real/models/item_within_benchmark/fusion_5814.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['with_retrieval']:raise ValueError('This inference entry point supports fusion without retrieval only')
        self.vocab=state['vocab'];self.temperature=state['temperature']
        self.network=Fusion(self.vocab,False).eval();self.network.load_state_dict(state['state_dict'])
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/minilm',local_files_only=True)
        self.encoder=AutoModel.from_pretrained(ROOT/'data/minilm',local_files_only=True).eval()
    def predict(self,record,calibrated=True):
        if not isinstance(record.get('item_text'),str) or not record['item_text'].strip():
            raise ValueError('A nonempty item_text string is required')
        ids=self.tokenizer.encode(record['item_text'],add_special_tokens=False,truncation=False)
        if len(ids)>254:ids=ids[:127]+ids[-127:]
        batch=self.tokenizer.pad([{'input_ids':[self.tokenizer.cls_token_id]+ids+[self.tokenizer.sep_token_id]}],return_tensors='pt')
        categories=torch.tensor([[self.vocab[c].get(record.get(c,''),0) for c in CATS]])
        with torch.inference_mode():
            h=self.encoder(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            x=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),p=2,dim=1)
            z=self.network(x,categories,torch.zeros(1)).item()
        p=float(expit(z/(self.temperature if calibrated else 1)))
        assert np.isfinite(p) and 0<=p<=1
        return p

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);args=parser.parse_args()
    print(json.dumps({'probability_correct':Predictor(args.checkpoint).predict(json.loads(args.input.read_text())),
        'scope':'Local research predictor; not an official competition submission'},indent=2))
