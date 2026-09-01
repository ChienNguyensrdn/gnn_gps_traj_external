import unittest
import numpy as np
from hybrid.rq2_data_only import normalize, probability_rows

class RQ2DataOnlyTests(unittest.TestCase):
 def test_priors_are_normalized_and_dbn_uses_transition(self):
  examples=[([0],[0],0,1,1)]; global_p=np.array([.5,.5]); users={0:np.array([9.,1.])}; times={1:np.array([1.,9.])}; trans={0:np.array([1.,9.])}
  rows=probability_rows(examples,global_p,users,times,trans,1.0)
  self.assertTrue(all(np.isclose(value.sum(),1) for value in rows.values())); self.assertGreater(rows["dbn-data-only"][0,1],rows["bn-data-only"][0,1])
 def test_normalize(self): self.assertTrue(np.allclose(normalize([1,3]),[.25,.75]))
if __name__=="__main__": unittest.main()
