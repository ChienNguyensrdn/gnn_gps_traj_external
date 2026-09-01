import unittest

from hybrid.all_cities_summary import aggregate, summarize_seeds


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


if __name__ == "__main__": unittest.main()
