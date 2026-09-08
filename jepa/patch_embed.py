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

    batch_size = ops.shape(images)[0]  # batch dim
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    gr, gc = h // patch_size, w // patch_size

    transpose_order = [0, 1, 3, 2, 4, 5]                              # B, gr, gc, py, px, C
    split_shape = [batch_size, gr, patch_size, gc, patch_size, c]     # 6 axes
    final_shape = [batch_size, gr * gc, patch_size * patch_size * c]  # B, 64, 48

    x = ops.reshape(images, split_shape)      # 1. split the axes
    x = ops.transpose(x, transpose_order)     # 2. actually move the numbers
    return ops.reshape(x, final_shape)        # 3. flatten


# ---------------------------------------------------------------------------
# Step 3 -- telling the model where each patch came from
# ---------------------------------------------------------------------------

def get_1d_sincos(d, pos):
    """Sine-cosine encoding of a 1-D sequence of positions.

    Args:
        d: width of the vector produced for each position. Must be even --
            half the dimensions come from sin, half from cos.
        pos: 1-D array of positions to encode, shape (M,).

    Returns:
        np.ndarray, shape (M, d).
    """
    # d // 2 frequencies, geometrically spaced from 1 down to 1e-4. Fast
    # "clocks" separate neighbouring positions, slow ones distant positions.
    omega = np.arange(d // 2, dtype=np.float32)
    omega /= d / 2
    omega = 1. / (10000 ** omega)  # (d/2,)

    # Outer product: every position seen by every frequency.
    angles = pos[:, None] * omega  # (M, d/2)

    # sin AND cos of the same angles. The pair is unambiguous -- a sine alone
    # takes each value twice per period -- and it turns a shift in position
    # into a fixed rotation of the encoding, which is what lets the predictor
    # reason about relative positions.
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


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
    # Row-major coordinates: the row varies slowly, the column varies fast.
    #   rows = 0,0,...,0, 1,1,...,1, ...   cols = 0,1,...,7, 0,1,...,7, ...
    # This is the order extract_patches emits its tokens in.
    coords = np.arange(grid_size, dtype=np.float32)
    rows = np.repeat(coords, grid_size)
    cols = np.tile(coords, grid_size)

    # Each half of the dimension budget encodes one coordinate.
    emb_row = get_1d_sincos(embed_dim // 2, rows)   # (grid_size**2, embed_dim//2)
    emb_col = get_1d_sincos(embed_dim // 2, cols)   # (grid_size**2, embed_dim//2)

    # axis=1: concatenate DIMENSIONS, not positions.
    return np.concatenate([emb_row, emb_col], axis=1)


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
