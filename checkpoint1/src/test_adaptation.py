import unittest
import numpy as np
import pandas as pd
from analyze_real_results import acquisition_split,adapt

class AdaptationTests(unittest.TestCase):
    def test_pools_disjoint_and_reproducible(self):
        frame=pd.DataFrame(dict(source=['test']*100,system_key=['s']*100,
            text_group=[str(i) for i in range(100)],label=[i%2 for i in range(100)]))
        a,e=acquisition_split(frame);a2,e2=acquisition_split(frame)
        self.assertTrue(set(a.text_group).isdisjoint(e.text_group))
        self.assertEqual(len(a),50);self.assertEqual(len(e),50)
        self.assertEqual(a.text_group.tolist(),a2.text_group.tolist())
        self.assertEqual(e.text_group.tolist(),e2.text_group.tolist())
        for small,large in [(1,3),(3,7),(7,15),(15,31)]:
            self.assertTrue(set(a.text_group.iloc[:small]).issubset(a.text_group.iloc[:large]))
    def test_zero_budget_identity(self):
        base=np.array([.2,.6,.9]);np.testing.assert_array_equal(adapt(base,[],[]),base)
    def test_successes_increase_probability(self):
        p=adapt(np.array([.3,.5]),np.array([.4]*7),np.ones(7))
        self.assertTrue((p>np.array([.3,.5])).all())
        self.assertTrue(((p>=0)&(p<=1)).all())

if __name__=='__main__':unittest.main(verbosity=2)
