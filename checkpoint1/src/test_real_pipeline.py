"""Checks of the real-data experiment's scientific boundaries."""
import unittest
import numpy as np
import pandas as pd
import torch
from prepare_real_data import binary_math,normalized_hash
from train_real_models import partitions,category_features,retrieval,Fusion,measures,CATS,ROOT

class RealPipelineTests(unittest.TestCase):
    def test_math_rubric_selection(self):
        self.assertTrue(binary_math('{"response_scale":{"values":[0,1]}}'))
        self.assertFalse(binary_math('{"response_scale":{"values":[0,0.5,1]}}'))
        self.assertFalse(binary_math('{"reference_answer":"yes"}'))
    def test_text_group_normalization(self):
        self.assertEqual(normalized_hash('HELLO  world\n'),normalized_hash('hello world'))
    def test_actual_split_isolation(self):
        df=pd.read_parquet(ROOT/'data/real_pilot.parquet')
        parts=partitions(df)
        self.assertEqual(len(parts),6)
        self.assertEqual(sum(len(p['test']) for mode,_,p in parts if mode=='benchmark'),len(df))
    def test_real_predictor_schema_excludes_answers(self):
        df=pd.read_parquet(ROOT/'data/real_pilot.parquet')
        for field in ['grading_criterion','reference_answer','verifier','trace']:
            self.assertNotIn(field,df.columns)
        self.assertTrue(df.label.isin([0,1]).all())
    def test_heldout_labels_cannot_change_retrieval(self):
        df=pd.DataFrame(dict(text_group=['a','b','c','d'],label=[0,1,1,0]))
        x=np.array([[1,0],[0,1],[.7,.7],[.8,.6]],dtype=np.float32)
        before=retrieval(df,x,np.array([0,1]))
        df.loc[[2,3],'label']=[0,1]
        np.testing.assert_allclose(before,retrieval(df,x,np.array([0,1])))
    def test_vocabulary_and_unseen_categories(self):
        df=pd.DataFrame([{c:'seen' for c in CATS},{c:'unseen' for c in CATS}])
        vocab,codes=category_features(df,np.array([0]))
        self.assertTrue((codes[1]==0).all())
        net=Fusion(vocab,True).eval()
        with torch.inference_mode():p=net(torch.zeros((2,384)),torch.tensor(codes),torch.full((2,),.5))
        self.assertTrue(torch.isfinite(p).all())
    def test_pair_macro_weighting(self):
        m=measures(np.array([0,0,0,1]),np.zeros(4),np.array(['a','a','a','b']))
        self.assertAlmostEqual(m['brier'],.25)
        self.assertAlmostEqual(m['macro_pair_brier'],.5)

if __name__=='__main__':unittest.main(verbosity=2)
