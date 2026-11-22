import json
import unittest
import numpy as np

from diagnostics.week35_selective_replay import make_stripe_scorer, replay


def fixture(n=30):
    """5 legal shared-wrong rows, 8 disagreement rows, all clean equal correct."""
    q=np.arange(1,n+1); g=np.r_[q,999]
    def both(mask):
        bb,fu=(np.full((n,n+1),-.2) for _ in range(2))
        for i in range(n):
            bb[i,i]=fu[i,i]=.80
            bb[i,-1]=fu[i,-1]=.77
            if mask and i<5:
                bb[i,-1]=fu[i,-1]=.82
            elif 5<=i<13:
                bb[i,i],bb[i,-1]=.78,.79
                fu[i,i],fu[i,-1]=.90,.80
        return dict(backbone=bb,fusion=fu,equal=.5*(bb+fu))
    return both(False),both(True),q,np.ones(n,dtype=int),g,np.full(n+1,2)


def good(q,indices):
    return np.array([.8 if i==q else .5 for i in indices])


class TestSelective(unittest.TestCase):
    def test_calibration_boundary_and_cheap_no_go_does_not_score(self):
        fx=fixture()
        with self.assertRaisesRegex(ValueError,"120"):
            replay(*fx)
        with self.assertRaisesRegex(ValueError,"forbidden"):
            replay(*fx,role="final")
        c,h,*ids=fx
        calls=[]
        def detector(*a):
            calls.append(a)
            raise AssertionError("No local computation when cheap gate fails")
        failed=replay(c,c,*ids,role="synthetic",clean_local=detector,heavy_local=detector)
        # Fails on lack of legal shared-wrong cases
        if failed["status"]!="NO_GO_PHASE0_CHEAP_BOUND":
            self.fail("Wrong high-quality fixture for cheap-bound test")
        self.assertFalse(calls)

    def test_operator_and_matched_controls(self):
        fx=fixture()
        bound=replay(*fx,role="synthetic")
        self.assertEqual(bound["status"],"AWAITING_FROZEN_LOCAL_EVIDENCE")
        self.assertEqual(bound["oracle"]["5"]["shared_wrong_with_legal_positive"],5)
        self.assertEqual(bound["activation_budget"],6)
        d=replay(*fx,clean_local=good,heavy_local=good,role="synthetic")
        self.assertEqual(d["status"],"CALIBRATION_ONLY_DO_NOT_OPEN_FINAL")
        self.assertEqual(d["local_operator_probe"]["actual_heavy_rescues_legal_shared_wrong"],5)
        self.assertFalse(d["demand_model"]["fit"])
        self.assertFalse(d["final_scope_opened"])
        self.assertEqual(d["matched_actual_activations"],{"clean":6,"heavy":6})
        self.assertEqual(d["matched_replay"]["disagreement_only"]["heavy"]["local"]["shared_wrong_rescues"],0)
        self.assertEqual(d["matched_replay"]["margin_only"]["heavy"]["rank_only"]["shared_wrong_rescues"],0)
        self.assertEqual(d,replay(*fx,clean_local=good,heavy_local=good,role="synthetic"))
        self.assertNotIn("indices", json.dumps(d))

    def test_gap_and_threshold_gate(self):
        fx=fixture()
        c,h,*ids=fx
        wide={k:x.copy() for k,x in h.items()}
        for k in wide:
            for i in range(5):wide[k][i,i]=.60
        d=replay(c,wide,*ids,role="synthetic",clean_local=good,heavy_local=good)
        self.assertEqual(d["status"],"NO_GO_PHASE0_LOCAL_HEADROOM")
        self.assertEqual(d["oracle"]["5"]["shared_wrong_with_legal_positive"],5)
        self.assertEqual(d["local_operator_probe"]["scorable_legal_shared_wrong"],0)
        self.assertNotIn("demand_model",d)

    def test_junk_is_not_legal_wrong_agreement_or_operational_positive(self):
        c,h,q,cam,g,gcam=fixture()
        g=g.copy();g[-1]=-1
        out=replay(c,h,q,cam,g,gcam,role="synthetic")
        self.assertEqual(out["status"],"NO_GO_PHASE0_CHEAP_BOUND")
        self.assertEqual(out["oracle"]["5"]["shared_wrong_with_legal_positive"],0)
        self.assertEqual(out["filtered_shared_raw_incumbents"]["heavy"],5)

    def test_stripe_score_best_two_and_unscorable(self):
        q=np.array([[[1,0],[1,0],[1,0],[1,0]]],dtype=float)
        g=np.array([[[1,0],[0,1],[-1,0],[1,0]],
                    [[0,0],[np.nan,2],[0,0],[1,0]],
                    [[1,0],[0,1],[-1,0],[-1,0]]],dtype=float)
        out=make_stripe_scorer(q,g)(0,(2,1,0))
        self.assertAlmostEqual(out[0],.5)
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2],1)

    def test_identity_oof_only_when_positive_and_negative_events_are_supported(self):
        c,h,*ids=fixture(120)
        for i in range(14,21):
            h['backbone'][i,-1]=h['fusion'][i,-1]=h['equal'][i,-1]=.82
        def harmful_heavy(q,indices):
            if 30<=q<37:
                return np.array([.5 if i==q else .85 for i in indices])
            return good(q,indices)
        d=replay(c,h,*ids,clean_local=good,heavy_local=harmful_heavy)
        self.assertTrue(d['demand_model']['fit'])
        self.assertEqual(d['demand_model']['folds'],5)
        self.assertIn('expected_gain',d['matched_replay'])
        self.assertFalse(d['independent_evaluation_authorized'])

    def test_real_shape_and_frozen_cost(self):
        fx=fixture(120)
        d=replay(*fx)
        self.assertEqual(d["activation_budget"],24)
        self.assertEqual(d["status"],"AWAITING_FROZEN_LOCAL_EVIDENCE")

if __name__=="__main__":
    unittest.main()
