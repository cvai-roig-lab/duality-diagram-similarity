# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:45:42 2020

@author: kshitij
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


def get_similarity_from_rdms(x,y,dist,feature_norm,debiased=True,centered=True):
    """
    Parameters
    ----------
    x : numpy matrix with dimensions n x p 
        task 1 features (n = number of images, p = feature dimensions) 
    y : numpy matrix with dimensions n x p
        task 1 features (n = number of images, p = feature dimensions) 
    dist : string
        distance function to compute dissimilarity matrices
    feature_norm : string
        feature normalization type
    debiased : bool, optional
        set True to perform unbiased centering 
    centered : bool, optional
        set True to perform unbiased centering 

    Returns
    -------
    DDS: float
        DDS between task1 and task2 

    """
    if feature_norm == 'None':
        x = x
        y = y
    elif feature_norm == 'centering':
        x = (x - np.mean(x,axis = 0, keepdims=True))
        y = (y - np.mean(y,axis = 0, keepdims=True))
    elif feature_norm == 'znorm':
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)
    elif feature_norm == 'group_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = group_norm(x,group_size=2)
        y = group_norm(y,group_size=2)
    elif feature_norm == 'instance_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = group_norm(x,group_size=8)
        y = group_norm(y,group_size=8)
    elif feature_norm == 'layer_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = group_norm(x,group_size=1)
        y = group_norm(y,group_size=1)
    elif feature_norm == 'batch_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = batch_norm(x)
        y = batch_norm(y)
    
    return cka(rdm(x,dist),rdm(y,dist),debiased=debiased,centered=centered)

def get_similarity(x,y,kernel,feature_norm,debiased=True,centered=True):
    """
    Parameters
    ----------
    x : numpy matrix with dimensions n x p 
        task 1 features (n = number of images, p = feature dimensions) 
    y : numpy matrix with dimensions n x p
        task 1 features (n = number of images, p = feature dimensions) 
    kernel : string
        kernel function to compute similarity matrices
    feature_norm : string
        feature normalization type
    debiased : bool, optional
        set True to perform unbiased centering 
    centered : bool, optional
        set True to perform unbiased centering 

    Returns
    -------
    DDS: float
        DDS between task1 and task2 

    """
    if feature_norm == 'None':
        x = x
        y = y
    elif feature_norm == 'centering':
        x = (x - np.mean(x,axis=0))
        y = (y - np.mean(y,axis=0))
    elif feature_norm == 'znorm':
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)
    elif feature_norm == 'group_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = group_norm(x,group_size=2)
        y = group_norm(y,group_size=2)
    elif feature_norm == 'instance_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = group_norm(x,group_size=8)
        y = group_norm(y,group_size=8)
    elif feature_norm == 'layer_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = group_norm(x,group_size=1)
        y = group_norm(y,group_size=1)
    elif feature_norm == 'batch_norm':
        x = np.reshape(x,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        y = np.reshape(y,(-1,16,16,8))      # reshaping features back to n x h x w x c format from flattened features
        x = batch_norm(x)
        y = batch_norm(y)

    if kernel == 'linear':
        return cka(gram_linear(x),gram_linear(y),debiased=debiased,centered=centered)
    elif kernel == 'rbf':
        return cka(gram_rbf(x, 0.5),gram_rbf(y, 0.5),debiased=debiased,centered=centered)
    elif kernel == 'lap':
        return cka(gram_laplacian_scipy(x),gram_laplacian_scipy(y),debiased=debiased,centered=centered)




def group_norm(x,group_size=2):
    """
    Parameters
    ----------
    x : numpy matrix with dimensions n x h x w x c 
        features (n = number of images, h = height, w = width, c = channel dimensions) 
    group_size : int, optional
        group size for group norm. The default is 2.
        group size (G)  = channel_dim (C) for instance normalization
        group size (G)  = 1 for layer normalization

    Returns
    -------
    normalized_x : numpy matrix with dimensions n x h*w*c
        normalized features

    """

    eps = 1e-9
    N, H, W, C = x.shape
    #print("The shape of features are ",N, H, W, C)
    G = group_size
    x = np.reshape(x, (N, H, W,G, C // G ))
    mean = np.mean(x,axis = (1, 2, 4), keepdims=True)
    var = np.var(x,axis = (1, 2, 4), keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    normalized_x = np.reshape(x, (N, C*H*W))

    return normalized_x

def batch_norm(x):
    """
    Parameters
    ----------
    x : numpy matrix with dimensions n x h x w x c 
        task 1 features (n = number of images, h = height, w = width, c = channel dimensions) 

    Returns
    -------
    normalized_x : numpy matrix with dimensions n x h*w*c
        normalized features

    """
    eps = 1e-9
    N, H, W, C = x.shape
    #print("The shape of features are ",N, H, W, C)
    mean = np.mean(x,axis = (0, 1, 2), keepdims=True)
    var = np.var(x,axis = (0, 1, 2), keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    normalized_x = np.reshape(x, (N, C*H*W))

    return normalized_x

def rdm(activations_value,dist):
    """

    Parameters
    ----------
    activations_value : numpy matrix with dimensions n x p 
        task 1 features (n = number of images, p = feature dimensions) 
    dist : string
        distance function to compute dissimilarity matrix

    Returns
    -------
    RDM : numpy matrix with dimensions n x n 
        dissimilarity matrices

    """
    if dist == 'pearson':
        RDM = 1-np.corrcoef(activations_value)
    elif dist == 'euclidean':
        RDM = euclidean_distances(activations_value)
    elif dist == 'cosine':
        RDM = 1- cosine_similarity(activations_value)
    return RDM


def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
    
  P.S. Function from Kornblith et al., ICML 2019

  """
  return x.dot(x.T)

def gram_laplacian_scipy(x):
  """Compute Gram (kernel) matrix for a laplacian kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  K = laplacian_kernel(x)
  return K

def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.

   P.S. Function from Kornblith et al., ICML 2019
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))



def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
   P.S. Function from Kornblith et al., ICML 2019
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False,centered=True):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
   P.S. Function from Kornblith et al., ICML 2019
  """
  if centered:
      gram_x = center_gram(gram_x, unbiased=debiased)
      gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)