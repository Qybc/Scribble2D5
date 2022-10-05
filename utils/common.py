"""Utilility function for all.
"""
# This code is borrowed and re-implemented from:
# https://github.com/jyhjinghwang/SegSort/blob/master/network/segsort/vis_utils.py
# https://github.com/jyhjinghwang/SegSort/blob/master/network/segsort/common_utils.py

import torch
import torch.nn.functional as F




def calculate_principal_components(embeddings, num_components=3):
  """Calculates the principal components given the embedding features.

  Args:
    embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
    num_components: An integer indicates the number of principal
      components to return.

  Returns:
    A 2-D float tensor of shape `[num_pixels, num_components]`.
  """
  embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
  _, _, v = torch.svd(embeddings)
  return v[:, :num_components]


def pca(embeddings, num_components=3, principal_components=None):
  """Conducts principal component analysis on the embedding features.

  This function is used to reduce the dimensionality of the embedding.

  Args:
    embeddings: An N-D float tensor with shape with the 
      last dimension as `embedding_dim`.
    num_components: The number of principal components.
    principal_components: A 2-D float tensor used to convert the
      embedding features to PCA'ed space, also known as the U matrix
      from SVD. If not given, this function will calculate the
      principal_components given inputs.

  Returns:
    A N-D float tensor with the last dimension as  `num_components`.
  """
  shape = embeddings.shape
  embeddings = embeddings.view(-1, shape[-1])

  if principal_components is None:
    principal_components = calculate_principal_components(
        embeddings, num_components)
  embeddings = torch.mm(embeddings, principal_components)

  new_shape = list(shape[:-1]) + [num_components]
  embeddings = embeddings.view(new_shape)

  return embeddings


def normalize_embedding(embeddings, eps=1e-12):
  """Normalizes embedding by L2 norm.

  This function is used to normalize embedding so that the
  embedding features lie on a unit hypersphere.

  Args:
    embeddings: An N-D float tensor with feature embedding in
      the last dimension.

  Returns:
    An N-D float tensor with the same shape as input embedding
    with feature embedding normalized by L2 norm in the last
    dimension.
  """
  norm = torch.norm(embeddings, dim=-1, keepdim=True)
  norm = torch.where(torch.ge(norm, eps),
                     norm,
                     torch.ones_like(norm).mul_(eps))
  return embeddings / norm

