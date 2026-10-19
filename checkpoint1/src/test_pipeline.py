"""Meaningful checks for gradients, leakage, metrics, and unknown systems."""
import unittest
import numpy as np
from run_experiments import Features, Network, metrics, sigmoid, split, synthetic, validate

class PipelineTests(unittest.TestCase):
    def test_benchmark_and_item_isolation(self):
        parts,_=split(validate(synthetic()),5814)
        for a in parts:
            for b in parts:
                if a!=b:
                    self.assertTrue({r['benchmark_id'] for r in parts[a]}.isdisjoint({r['benchmark_id'] for r in parts[b]}))
                    self.assertTrue({r['item_id'] for r in parts[a]}.isdisjoint({r['item_id'] for r in parts[b]}))
    def test_duplicate_text_rejected(self):
        rows=synthetic()[:2]
        rows[1]['benchmark_id']='other'
        with self.assertRaises(ValueError): validate(rows)
    def test_label_rejected(self):
        rows=synthetic()[:1]; rows[0]['label']=2
        with self.assertRaises(ValueError): validate(rows)
    def test_metrics(self):
        y=np.array([0,1,0,1]); p=np.full(4,.5)
        self.assertAlmostEqual(metrics(y,p)['brier'],.25)
        self.assertAlmostEqual(metrics(y,p)['log_loss'],np.log(2))
        with self.assertRaises(ValueError): metrics(y,np.full(4,np.nan))
    def test_unknown_category(self):
        rows=synthetic()[:1]; f=Features(rows)
        rows[0]=dict(rows[0],subject_id='unseen',provider='new')
        _,m=f.transform(rows)
        self.assertEqual(m[0,0],1)
        self.assertEqual(m.sum(),2)
    def test_gradients_finite_difference(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(6,3)); y=np.array([0,1,0,1,1,0])
        net=Network([3,4,2,1],1)
        dw,db=net.gradients(x,y,decay=0)
        def loss():
            z=net.forward(x)[0]
            return np.mean(np.logaddexp(0,z)-y*z)
        for param,grad in zip(net.w+net.b,dw+db):
            idx=tuple(0 for _ in param.shape); orig=param[idx]; eps=1e-5
            param[idx]=orig+eps; plus=loss()
            param[idx]=orig-eps; minus=loss()
            param[idx]=orig
            self.assertAlmostEqual(grad[idx],(plus-minus)/(2*eps),places=6)
    def test_training_reduces_loss(self):
        x=np.array([[-2.],[-1.],[1.],[2.]]); y=np.array([0,0,1,1])
        net=Network([1,4,1],12)
        before=metrics(y,sigmoid(net.forward(x)[0]))['log_loss']
        net.fit(x,y,x,y,epochs=100)
        self.assertLess(metrics(y,sigmoid(net.forward(x)[0]))['log_loss'],before)

if __name__=='__main__': unittest.main(verbosity=2)
