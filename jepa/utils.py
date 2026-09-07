"""Small infrastructure helpers: reproducibility, parameter counts, schedules.

Nothing here is specific to JEPA -- these are the generic pieces any training
script needs. The JEPA-specific logic lives in the other modules.
"""

import os
import random

import numpy as np
import keras
from keras import ops


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and the Keras backend so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def count_params(model: keras.Model) -> tuple:
    """Return (trainable, non_trainable) parameter counts for a built model."""
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return trainable, non_trainable


def format_params(n: int) -> str:
    """4823296 -> '4.82M'. Just for readable log lines."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class CosineWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup, then cosine decay -- the standard ViT recipe.

    Transformers are unstable in the first few hundred steps, when attention
    weights are still random. Ramping the learning rate up from ~0 avoids the
    early divergence you otherwise get, and the cosine tail anneals it smoothly.
    """

    def __init__(self, base_lr, total_steps, warmup_steps, final_lr_frac=0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_lr_frac = float(final_lr_frac)

    def __call__(self, step):
        step = ops.cast(step, "float32")
        warmup = max(float(self.warmup_steps), 1.0)

        # Phase 1: ramp 0 -> base_lr over the warmup steps.
        warm_lr = self.base_lr * step / warmup

        # Phase 2: cosine from base_lr down to base_lr * final_lr_frac.
        decay_steps = max(float(self.total_steps - self.warmup_steps), 1.0)
        progress = ops.clip((step - warmup) / decay_steps, 0.0, 1.0)
        cosine = 0.5 * (1.0 + ops.cos(np.pi * progress))
        cos_lr = self.base_lr * (self.final_lr_frac + (1 - self.final_lr_frac) * cosine)

        return ops.where(step < warmup, warm_lr, cos_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "final_lr_frac": self.final_lr_frac,
        }


def ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    """Momentum for the target encoder's EMA update, annealed start -> end.

    Early on the target encoder should track the context encoder fairly quickly
    (lower momentum); later it should be almost frozen (momentum -> 1.0) so the
    prediction targets stop drifting and the loss can actually converge.
    """
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return end - (end - start) * (1.0 + np.cos(np.pi * progress)) / 2.0
