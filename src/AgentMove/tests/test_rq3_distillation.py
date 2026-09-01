import unittest

import numpy as np
import pandas as pd

from hybrid.rq3_distillation import fit_bn_statistics, fuse_sources, platt_likelihood_ratio, select_weights


class RQ3DistillationTests(unittest.TestCase):
    def test_zero_weights_reproduce_bn(self):
        bn=np.array([[.7,.3]]); q=np.array([[.2,.8]]); llm=np.array([[.5,2.]])
        self.assertTrue(np.allclose(fuse_sources(bn,q,llm,0,0),bn))

    def test_positive_teacher_weight_can_change_ranking(self):
        bn=np.array([[.6,.4]]); q=np.array([[.1,.9]]); llm=np.ones((1,2))
        self.assertEqual(int(np.argmax(fuse_sources(bn,q,llm,1,0))),1)

    def test_likelihood_ratio_removes_base_odds(self):
        config={"alpha":1.0,"beta":0.0,"prevalence":0.5,"max_ratio":20.0}
        values=platt_likelihood_ratio([0.0],config)
        self.assertTrue(np.allclose(values,[1.0]))

    def test_validation_selection_never_uses_test(self):
        bn=np.array([[.6,.4],[.6,.4]]); q=np.array([[.1,.9],[.1,.9]])
        llm=np.ones_like(bn); labels=np.array([1,1])
        selected,_=select_weights(bn,q,llm,labels,[0.0,1.0])
        self.assertEqual(selected["M3-quantitative"]["quantitative"],1.0)

    def test_bn_statistics_use_encoded_poi_column_indices(self):
        frame=pd.DataFrame({"POI_id":[0,1,1],"user_id":["u","u","u"],
                            "UTC_time":["2020-01-01 00:00:00"]*3})
        global_prior,users,_,matched=fit_bn_statistics(frame,2,1.0)
        self.assertEqual(matched,3); self.assertGreater(global_prior[1],global_prior[0])
        self.assertEqual(users["u"][1],2)


if __name__ == "__main__": unittest.main()
