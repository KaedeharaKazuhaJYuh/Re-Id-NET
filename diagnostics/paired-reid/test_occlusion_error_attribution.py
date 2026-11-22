import unittest
import numpy as np
from diagnostics.occlusion_error_attribution import BUCKETS, decompose

class TestAttribution(unittest.TestCase):
    def test_partition_and_filters(self):
        qids=[1,1,1,1,-1,2]
        qc=[2]*6
        gid=[1,1,-1,3,4,5,6,7,8]
        gc=[2,3,2,3,3,3,3,3,3]
        def make(ranks):
            a=[]
            for rank in [*ranks,1,1]:
                a.append([100,7.5-rank,99,6,5,4,3,2,1])
            return np.array(a,dtype=float)
        prev=dict(backbone=make([1,1,2,6]),fusion=make([2,2,4,7]),equal=make([1]*4))
        late=dict(backbone=make([1,3,6,6]),fusion=make([2,4,7,7]),equal=make([2]*4))
        z=decompose(prev,late,qids,qc,gid,gc)
        self.assertEqual(z["valid_queries"],4)
        self.assertEqual(z["new_equal_rank1_misses"],4)
        self.assertEqual(z["buckets"],dict.fromkeys(BUCKETS,1))
        self.assertEqual(z["transitions"]["backbone"]["topk_1_to_0"],1)
        x=z["mean_positive_minus_negative"][BUCKETS[0]]["equal"]
        self.assertEqual(x,dict(n=1,clean_mean=.5,masked_mean=-.5,delta_mean=-1.0))
    def test_no_new_miss_and_validation(self):
        x=np.array([[1,0]])
        group=dict.fromkeys(("backbone","fusion","equal"),x)
        r=decompose(group,group,[1],[2],[1,2],[3,3])
        self.assertEqual(r["new_equal_rank1_misses"],0)
        malformed=dict(group);malformed["backbone"]=np.array([[np.nan,0]])
        with self.assertRaises(ValueError):
            decompose(group,malformed,[1],[2],[1,2],[3,3])

if __name__=="__main__":
    unittest.main()
