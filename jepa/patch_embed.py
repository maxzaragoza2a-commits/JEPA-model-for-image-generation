"""Patch embedding: an image becomes a sequence of token vectors.

This is the doorway between "picture" and "transformer". A transformer has no
idea what an image is -- it consumes a *sequence of vectors*. So we cut the
image into a grid of small square patches and turn each patch into one vector.
Those vectors are the "words"; the image is the "sentence".

The shape journey, for our config (32x32x3 image, 4x4 patches, embed_dim=128):

    (B, 32, 32, 3)     a batch of images
         |  step 1: cut into a grid of non-overlapping patches, then flatten each
    (B, 64, 48)        64 patches per image, each 4*4*3 = 48 raw numbers
         |  step 2: project every patch through ONE shared Dense layer
    (B, 64, 128)       64 tokens, each a learned 128-dim embedding
         |  step 3: add a positional encoding
    (B, 64, 128)       tokens that also know WHERE in the image they came from

Step 3 is not optional. Self-attention is permutation-invariant: shuffle the
tokens and it computes the same thing. Without positional information the model
literally cannot tell a face from the same patches scrambled.
"""

import numpy as np
import keras
from keras import ops


# ---------------------------------------------------------------------------
# Step 1 -- cutting the image into patches
# ---------------------------------------------------------------------------
def extract_patches(images, patch_size: int):
    """Cut a batch of images into a flat sequence of raw patch vectors.

    A plain reshape does NOT work here: reshape re-groups numbers in memory
    order, and memory order runs along whole rows of the image, whereas a 4x4
    patch is not contiguous in memory. The trick is to split each spatial axis
    into (which patch, where inside the patch), transpose so the two "which
    patch" axes are adjacent, then merge.

        (B, H, W, C)
          -> reshape   (B, H//p, p, W//p, p, C)
                        B   gr   py  gc   px  C     <- axis meanings
          -> transpose (B, gr, gc, py, px, C)
          -> reshape   (B, gr*gc, py*px*C)

    Args:
        images: float tensor, shape (B, H, W, C).
        patch_size: side length of each square patch. Must divide H and W.

    Returns:
        Tensor of shape (B, num_patches, patch_size * patch_size * C), where
        num_patches = (H // patch_size) * (W // patch_size).

        Patches are ordered ROW-MAJOR: left-to-right along the top row first,
        then the next row down. Everything downstream (positional encodings,
        masking) assumes this ordering, so it has to stay consistent.
    """
    b = ops.shape(images)[0]                # dynamic: unknown until runtime
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    gr, gc = h // patch_size, w // patch_size

    # Split H -> (gr, py) and W -> (gc, px), giving six axes:
    #   B, grid_row, row-in-patch, grid_col, col-in-patch, channels
    x = ops.reshape(images, [b, gr, patch_size, gc, patch_size, c])

    # Bring the two "which patch" axes next to each other, ahead of the two
    # "where inside the patch" axes. This is the step reshape alone cannot do.
    x = ops.transpose(x, [0, 1, 3, 2, 4, 5])

    # Merge (gr, gc) -> num_patches and (py, px, C) -> patch_dim.
    return ops.reshape(x, [b, gr * gc, patch_size * patch_size * c])


# ---------------------------------------------------------------------------
# Step 3 -- telling the model where each patch came from
# ---------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Fixed (non-learned) 2D sine-cosine positional encodings.

    Each grid cell (row, col) gets a unique `embed_dim`-length signature built
    from sines and cosines at many frequencies. Half the dimensions encode the
    row, half encode the column.

    We use fixed encodings rather than learned ones because the JEPA predictor
    must be queried at target positions it has to reason about generically;
    a smooth, analytic position code generalises better there than a lookup
    table of learned vectors.

    Args:
        embed_dim: width of each positional vector. Must be divisible by 4
            (split row/col, and each of those splits into sin/cos).
        grid_size: patches per side (8 for us).

    Returns:
        np.ndarray, shape (grid_size * grid_size, embed_dim), row-major to
        match `extract_patches`.
    """
    # TODO (together)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The layer that ties steps 1-3 together
# ---------------------------------------------------------------------------
class PatchEmbedding(keras.layers.Layer):
    """(B, H, W, C) images -> (B, num_patches, embed_dim) positioned tokens."""

    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # TODO (together): the projection layer and the positional table.

    def build(self, input_shape):
        # TODO (together): derive grid_size / num_patches from input_shape,
        # create the Dense projection, and store the positional encodings.
        raise NotImplementedError

    def call(self, images):
        # TODO (together): steps 1 -> 2 -> 3.
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config
