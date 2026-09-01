import unittest
import numpy as np
import tempfile
from collections import Counter
from pathlib import Path
from hybrid.rq2_data_only import normalize, probability_rows
from hybrid.rq2_aggregate import paired_comparisons

class RQ2DataOnlyTests(unittest.TestCase):
 def test_priors_are_normalized_and_dbn_uses_transition(self):
  examples=[([0],[0],0,1,1)]; global_p=np.array([.5,.5]); users={0:Counter({0:8})}; times={1:Counter({1:8})}; trans={0:Counter({1:8})}
  rows=probability_rows(examples,global_p,users,times,trans,1.0)
  self.assertTrue(all(np.isclose(value.sum(),1) for value in rows.values())); self.assertGreater(rows["dbn-data-only"][0,1],rows["bn-data-only"][0,1])
 def test_normalize(self): self.assertTrue(np.allclose(normalize([1,3]),[.25,.75]))
 def test_aggregate_includes_direct_dbn_vs_bn_without_pseudo_seeds(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory); seeds=[42,43,44]
   payload={"query_index":np.arange(4),"labels":np.array([0,1,2,3]),
            "ranks":np.array([1,2,6,11]),"reciprocal_rank":np.array([1,.5,.1,.05]),
            "true_probability":np.array([.8,.4,.2,.1]),"brier":np.array([.1,.3,.5,.7])}
   for variant in ("unigram","markov-bigram","bn-data-only","dbn-data-only"):
    path=root/variant/"seed-42"; path.mkdir(parents=True)
    changed={key:value.copy() for key,value in payload.items()}
    if variant != "dbn-data-only": changed["ranks"]=changed["ranks"]+1
    np.savez_compressed(path/"test.predictions.npz",**changed)
   for seed in seeds:
    path=root/"quantitative-teacher"/f"seed-{seed}"; path.mkdir(parents=True)
    np.savez_compressed(path/"test.predictions.npz",**payload)
   rows=paired_comparisons(root,seeds,1000)
   direct=[row for row in rows if row["comparison"]=="dbn-data-only-vs-bn-data-only"]
   self.assertEqual(len(direct),6)
   self.assertTrue(all(row["replicates"]==1 for row in direct))
   recall1=next(row for row in direct if row["metric"]=="recall@1")
   self.assertGreater(recall1["effect"],0)
if __name__=="__main__": unittest.main()
