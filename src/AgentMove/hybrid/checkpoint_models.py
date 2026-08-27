from __future__ import annotations

from .neural_cgm import ModelConfig, build_model as build_gru
from .transformer_teacher import TransformerConfig, build_model as build_transformer


def build_checkpoint_model(checkpoint):
    architecture = checkpoint.get("architecture", "gru")
    if architecture == "gru": return build_gru(ModelConfig(**checkpoint["config"])), ModelConfig(**checkpoint["config"])
    if architecture == "transformer":
        config = TransformerConfig(**checkpoint["config"]); return build_transformer(config), config
    raise ValueError(f"unsupported checkpoint architecture: {architecture}")
