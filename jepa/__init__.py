"""A from-scratch JEPA (Joint Embedding Predictive Architecture) for images.

See README.md for the project goal. Components are built up in this order:
    config        -- all hyperparameters in one place
    data          -- CIFAR-10 input pipeline
    patch_embed   -- image -> patches -> embeddings (+ positional encoding)
    attention     -- multi-head self-attention, written out by hand
    transformer   -- transformer block and the ViT encoder
    masking       -- I-JEPA style context/target block sampling
    model         -- context encoder + EMA target encoder + predictor
"""

from jepa.config import JEPAConfig

__all__ = ["JEPAConfig"]
