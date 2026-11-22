import unittest
import numpy as np
from diagnostics.occlusion_branch_factorial import analyze, CM, MC


def contract(cb,cf,mb,mf,query_ids=None,query_cams=None,gallery_ids=None,gallery_cams=None,condition='heavy'):
    cb,cf,mb,mf = map(lambda x:np.asarray(x,dtype=float),(cb,cf,mb,mf))
    clean={'backbone':cb,'fusion':cf,'equal':.5*(cb+cf)}
    mask={'backbone':mb,'fusion':mf,'equal':.5*(mb+mf)}
    mixed={CM:.5*(cb+mf),MC:.5*(mb+cf)}
    q,g=cb.shape
    args=(query_ids if query_ids is not None else [1]*q,
          query_cams if query_cams is not None else [2]*q,
          gallery_ids if gallery_ids is not None else [1]+list(range(2,g+1)),
          gallery_cams if gallery_cams is not None else [3]*g)
    return clean,mask,args,mixed,condition


def run(*inputs,**kw):
    a,b,ids,m,c=contract(*inputs,**kw)
    return analyze(a,b,*ids,mixed=m,condition=c)


class FactorialTests(unittest.TestCase):
    def test_restoration_partition_and_signed_margins(self):
        baseline=[[.6,0]]*4
        bb=[[.2,0],[-.8,0],[-.8,0],[-.1,0]]
        fu=[[-.8,0],[.2,0],[-.8,0],[-.1,0]]
        d=run(baseline,baseline,bb,fu)
        f=d['factorial']
        self.assertEqual(d['newly_wrong_equal'],4)
        self.assertTrue(d['primary'])
        self.assertEqual(f['restoration_partition_among_new_equal_misses'],
            {'only_restore_clean_backbone':1,'only_restore_clean_fusion':1,
             'both_single_restorations_work':1,'neither_single_restoration_works':1})
        self.assertEqual(f['restored_among_new_equal_misses'],{CM:2,MC:2})
        self.assertEqual(f['vs_masked_both'][CM]['rescues'],2)
        self.assertEqual(f['difference_in_differences']['rank1'],0)
        self.assertAlmostEqual(d['paired_masked_minus_clean_margin']['equal']['all']['mean'],-.975)
        self.assertEqual(d['raw_union']['5']['new_equal_miss_buckets']['filtered_branch_rank1_survived'],2)
        self.assertTrue(all(x==0 for x in d['native_mix_score_max_abs_diff'].values()))

    def test_raw_candidates_before_junk_same_cam_filter_and_stable_ties(self):
        # Invalid gallery entries win ties and occupy the raw shortlist.
        cb=[[1.0]*5+[.6,.4,.1]]
        cf=cb
        mb=[[1.0]*5+[.4,.6,.1]]
        mf=mb
        d=run(cb,cf,mb,mf,gallery_ids=[-1,1,-1,1,-1,1,2,3],
              gallery_cams=[99,2,99,2,99,3,3,3],condition='light')
        self.assertFalse(d['primary'])
        self.assertEqual(set(d['raw_union']),{'5','10','20'})
        self.assertEqual(d['newly_wrong_equal'],1)
        self.assertEqual(d['raw_union']['5']['valid_positive_transition']['0_to_0'],1)
        self.assertEqual(d['raw_union']['5']['mean_raw_and_valid_union_size']['clean'],[5,0])
        self.assertEqual(d['filtered_positive_rank']['5']['transitions']['backbone']['topk_1_to_1'],1)
        self.assertEqual(d['raw_union']['10']['valid_positive_transition']['1_to_1'],1)

    def test_mask_removes_raw_candidate_without_equating_with_no_valid_positive(self):
        c=[[.9]+[.4]*22]
        m=[[.3]+[.4]*9+[.1]*13]
        d=run(c,c,m,m)
        for k,key in [('5','1_to_0'),('10','1_to_1'),('20','1_to_1')]:
            self.assertEqual(d['raw_union'][k]['valid_positive_transition'][key],1)
        self.assertEqual(d['raw_union']['5']['new_equal_miss_buckets']['raw_shortlist_lost_valid_positive'],1)
        self.assertEqual(d['raw_union']['10']['new_equal_miss_buckets']['raw_shortlist_valid_positive_survived'],1)

    def test_same_query_disagreement_and_candidate_count(self):
        c=[[.9,.4,.1]]*2
        mb=[[.7,.8,.1],[.1,.8,.1]]
        mf=[[.2,.1,.8],[.1,.8,.1]]
        d=run(c,c,mb,mf)
        r=d['raw_union']['5']
        self.assertEqual(r["new_equal_miss_raw_trigger"]["masked_raw_top1_disagreement_and_positive_in_union"],1)
        self.assertEqual(r["raw_top1_disagreement"]["masked"]["raw_top1_disagreements"],1)
        self.assertEqual(r["raw_top1_disagreement"]["clean"]["raw_top1_disagreements_equal_correct"],0)
        groups=d['factorial']['margin_groups_by_filtered_k5']
        self.assertEqual(sum(x['count'] for x in groups.values()),2)
        self.assertEqual(sum(x['effects']['descriptive_interaction']['n'] for x in groups.values()),2)
        self.assertEqual(groups['branch_positive_rank2_to_k']['count'],2)

    def test_exact_input_used_at_near_tie_and_fail_closed(self):
        c,m,args,mix,_=contract([[.2,.2]],[[.2,.2]],[[.2,.2]],[[.2,.2]])
        mix[CM][0,1] += 1e-7    # score parity passes but changes exact stable ordering
        d=analyze(c,m,*args,mixed=mix,condition='heavy')
        self.assertEqual(d['factorial']['vs_masked_both'][CM]['harms'],1)
        mix[CM][0,1] += 1e-3
        with self.assertRaisesRegex(ValueError,"parity"):
            analyze(c,m,*args,mixed=mix,condition='heavy')

    def test_metadata_exclusions_and_no_negative_margin(self):
        zero=[[0,0]]*3
        d=run(zero,zero,zero,zero,query_ids=[1,-1,2],gallery_ids=[1,-1])
        self.assertEqual(d['valid_queries'],1)
        self.assertEqual(d['factorial']['margin']['clean_both']['all'],dict(n=0,mean=None,median=None))
        self.assertEqual(d['branch_margin']['clean']['backbone']['new_equal_misses'],dict(n=0,mean=None,median=None))

if __name__=='__main__':
    unittest.main()
