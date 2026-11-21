import unittest
import numpy as np

from fusion import ambiguity, fuse_scores, required_rows


class FusionTests(unittest.TestCase):
    def test_prefix_never_promotes_outside_document(self):
        q = np.array(['train:a']*4)
        d = np.array(['1','2','3','4'])
        lexical = np.array([4.,3.,2.,1.])
        neural = np.array([0.,1.,1e9,1e10])
        scores = fuse_scores(q,d,lexical,neural,np.ones(4,bool),
                             {'neural_weight':1.,'top_k':2})
        self.assertEqual(np.argsort(-scores).tolist(),[1,0,2,3])

    def test_gate_and_mask_are_query_local(self):
        q = np.array(['train:a']*3+['train:b']*3)
        d = np.array(['1','2','3']*2)
        lexical = np.array([10.,0.,0.,1.,1.,1.])
        config={'neural_weight':1.,'top_k':2,'entropy_threshold':.99}
        mask=required_rows(q,d,lexical,config)
        self.assertEqual(mask.tolist(),[False,False,False,True,True,False])
        neural=np.array([np.nan,np.nan,np.nan,0.,1.,np.nan])
        result=fuse_scores(q,d,lexical,neural,mask,config)
        self.assertEqual(np.argsort(-result[3:]).tolist(),[1,0,2])
        self.assertTrue(np.isfinite(result).all())

    def test_missing_required_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            fuse_scores(['train:a']*2,['1','2'],[2.,1.],[0.,0.],[True,False],
                        {'neural_weight':.5,'top_k':2})

    def test_zero_weight_preserves_scores(self):
        result=fuse_scores(['train:a']*2,['1','2'],[3.,-8.],[],[],{'neural_weight':0.})
        np.testing.assert_array_equal(result,[3.,-8.])

    def test_entropy_shift_and_scale_invariant(self):
        self.assertAlmostEqual(ambiguity(np.array([3.,2.,1.])),
                               ambiguity(np.array([130.,120.,110.])))
        self.assertAlmostEqual(ambiguity(np.ones(5)),1.)


if __name__=='__main__':
    unittest.main()
