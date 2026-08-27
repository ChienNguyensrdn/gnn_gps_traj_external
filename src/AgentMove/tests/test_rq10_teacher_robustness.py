import unittest

from hybrid.checkpoint_models import build_checkpoint_model
from hybrid.neural_cgm import ModelConfig
from hybrid.transformer_teacher import TransformerConfig, build_model


class RQ10TeacherRobustnessTests(unittest.TestCase):
    def test_transformer_exposes_matched_distillation_states(self):
        import torch
        config = TransformerConfig(num_pois=17, num_users=3, hidden_dim=16, heads=4, layers=2)
        model = build_model(config)
        output = model(
            torch.tensor([[1, 2, 0], [3, 4, 5]]),
            torch.tensor([[1, 2, 0], [2, 3, 4]]),
            torch.tensor([2, 3]), torch.tensor([0, 1]), torch.tensor([3, 4]),
            return_states=True,
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 17))
        self.assertEqual([tuple(value.shape) for value in output["depth_states"]], [(2, 16), (2, 16)])
        self.assertEqual(tuple(output["temporal_states"].shape), (2, 3, 16))
        self.assertTrue(torch.equal(output["temporal_states"][0, 2], torch.zeros(16)))

    def test_checkpoint_factory_supports_legacy_gru_and_transformer(self):
        gru_config = ModelConfig(num_pois=11, num_users=2, hidden_dim=16)
        gru, loaded_gru = build_checkpoint_model({"config": vars(gru_config)})
        transformer_config = TransformerConfig(num_pois=11, num_users=2, hidden_dim=16, heads=4)
        transformer, loaded_transformer = build_checkpoint_model(
            {"architecture": "transformer", "config": vars(transformer_config)})
        self.assertEqual(gru.head[-1].out_features, 11)
        self.assertEqual(transformer.head[-1].out_features, 11)
        self.assertEqual(loaded_gru.hidden_dim, loaded_transformer.hidden_dim)


if __name__ == "__main__":
    unittest.main()
