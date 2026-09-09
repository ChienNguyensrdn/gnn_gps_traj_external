import unittest

from hybrid.all_cities_summary import aggregate, partition_missing, summarize_seeds


class AllCitiesSummaryTests(unittest.TestCase):
    def test_macro_is_unweighted_and_city_variance_is_population(self):
        rows={"A":{"exp":{"recall@1":.1}},"B":{"exp":{"recall@1":.3}}}
        result=aggregate(["A","B"],rows)["exp"]["recall@1"]
        self.assertAlmostEqual(result["macro_mean"],.2)
        self.assertAlmostEqual(result["city_population_variance"],.01)

    def test_only_common_experiments_are_aggregated(self):
        rows={"A":{"common":{"mrr":.1},"partial":{"mrr":.2}},"B":{"common":{"mrr":.3}}}
        self.assertEqual(set(aggregate(["A","B"],rows)),{"common"})

    def test_seed_summary_reports_sample_standard_deviation(self):
        macro={"RQ/variant/seed-42":{"mrr":{"macro_mean":.1,"city_population_variance":.01}},
               "RQ/variant/seed-43":{"mrr":{"macro_mean":.3,"city_population_variance":.03}}}
        result=summarize_seeds(macro)["RQ/variant"]["mrr"]
        self.assertAlmostEqual(result["mean"],.2)
        self.assertAlmostEqual(result["std"],2 ** .5 / 10)
        self.assertAlmostEqual(result["city_population_variance_mean"],.02)

    def test_internal_gate_accepts_only_gpu_contention(self):
        missing={"Tokyo":["GPU_CONTENTION:path/a","missing/path/b"],
                 "Paris":["GPU_CONTENTION:path/c"]}
        blocking,accepted=partition_missing(missing,allow_gpu_contention=True)
        self.assertEqual(blocking,{"Tokyo":["missing/path/b"],"Paris":[]})
        self.assertEqual(accepted,{"Tokyo":["GPU_CONTENTION:path/a"],
                                   "Paris":["GPU_CONTENTION:path/c"]})

    def test_strict_gate_keeps_contention_blocking(self):
        missing={"Tokyo":["GPU_CONTENTION:path/a"]}
        blocking,accepted=partition_missing(missing,allow_gpu_contention=False)
        self.assertEqual(blocking,missing)
        self.assertEqual(accepted,{"Tokyo":[]})


if __name__ == "__main__": unittest.main()
