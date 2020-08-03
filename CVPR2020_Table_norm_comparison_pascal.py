import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import chi2_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,rbf_kernel
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import numpy as np
import scipy
import scipy.io as sio
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse
import time

from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics.pairwise import chi2_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats.stats import pearsonr,spearmanr
from sklearn.preprocessing import StandardScaler
from scipy import linalg,spatial
from sklearn.covariance import GraphicalLassoCV, LedoitWolf,OAS, ShrunkCovariance

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole pascal_voc_segmentation'

def group_norm(x,group_size=2):
    # group size (G)  = channel_dim (C) for instance normalization
    # group size (G)  = 1 for layer normalization
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
    eps = 1e-9
    N, H, W, C = x.shape
    #print("The shape of features are ",N, H, W, C)
    mean = np.mean(x,axis = (0, 1, 2), keepdims=True)
    var = np.var(x,axis = (0, 1, 2), keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    normalized_x = np.reshape(x, (N, C*H*W))

    return normalized_x

def z_norm_my_implementation(x):
    eps = 1e-9
    N, H, W, C = x.shape
    x = np.reshape(x, (N, C*H*W))
    mean = np.mean(x,axis = 0, keepdims=True)
    var = np.var(x,axis = 0, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    normalized_x = np.reshape(x, (N, C*H*W))

    return normalized_x

def covGL(x):
    eps = 1e-9
    N, H, W, C = x.shape
    x = np.reshape(x, (N, C*H*W))

    #x = StandardScaler().fit_transform(x)
    model = GraphicalLassoCV()
    model.fit(x)
    prec = np.sqrt(model.precision_)
    normalized_x = np.matmul(x,prec)
    return normalized_x

def covLF(x):
    eps = 1e-9
    N, H, W, C = x.shape
    x = np.reshape(x, (N, C*H*W))
    mean = np.mean(x,axis = 0, keepdims=True)
    x = x - mean
    #x = StandardScaler().fit_transform(x)
    start = time.time()
    lw= LedoitWolf(store_precision=True,assume_centered=True).fit(x)
    end = time.time()
    print("Time for covariance computation: ",end - start)
    start = time.time()
    lw_prec = scipy.linalg.fractional_matrix_power(lw.precision_, 0.5)
    end = time.time()
    print("Time for inverse sqrt computation: ",end - start)
    #print(lw_prec.min(),lw_prec.max(),lw_prec.mean())
    normalized_x = np.matmul(x,lw_prec)
    return normalized_x

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)

def gram_laplacian_scipy(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  K = laplacian_kernel(x)
  #print(K.shape)
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

def pearson(gram_x, gram_y,num_images = 200, debiased=False,centered=True):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    if centered:
      gram_x = center_gram(gram_x, unbiased=debiased)
      gram_y = center_gram(gram_y, unbiased=debiased)
    gram_x = gram_x[np.triu_indices(num_images,1)]
    gram_y = gram_y[np.triu_indices(num_images,1)]
    #print(taskonomy_rdm.shape)
    #pearson_correlation,_ = pearsonr(gram_x, gram_y)
    pearson_correlation = 1 - spatial.distance.cosine(gram_x, gram_y)
    return pearson_correlation


def spearman(gram_x, gram_y,num_images = 200, debiased=False,centered=True):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    if centered:
      gram_x = center_gram(gram_x, unbiased=debiased)
      gram_y = center_gram(gram_y, unbiased=debiased)
    gram_x = gram_x[np.triu_indices(num_images,1)]
    gram_y = gram_y[np.triu_indices(num_images,1)]
    #print(taskonomy_rdm.shape)
    spearman_correlation,_ = spearmanr(gram_x, gram_y)
    return spearman_correlation

def get_features(taskonomy_feats_path,pascal_feats_path):
    taskonomy_tasks = ['autoencoder','class_1000', 'class_places', 'colorization','curvature',\
                       'denoise', 'edge2d', 'edge3d', \
                       'inpainting_whole','keypoint2d', 'keypoint3d', \
                       'reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','room_layout' ,\
                       'segment25d', 'segment2d', 'segmentsemantic', 'vanishing_point']
    print(len(taskonomy_tasks))
    taskonomy_list={}
    print(taskonomy_feats_path)
    for task in taskonomy_tasks:
        taskonomy_list[task] = glob.glob(taskonomy_feats_path+"/*"+  task +"_encoder_output.npy")
        taskonomy_list[task].sort()
        print(task, len(taskonomy_list[task]))

    #Loading data
    num_images = len(taskonomy_list[task])
    print(np.load(taskonomy_list[task][0]).shape)
    a=np.load(taskonomy_list[task][0]).ravel()
    print(a.shape)
    num_features =a.shape[0]
    taskonomy_data = {}
    for task in taskonomy_tasks:
        taskonomy_data[task] = np.zeros((num_images,num_features))
        for i,taskonomy_file in enumerate(taskonomy_list[task]):
            taskonomy_data[task][i,:]  = np.load(taskonomy_file).ravel()

    pascal_list = glob.glob(pascal_feats_path+"*.npy")
    pascal_list.sort()
    print(len(pascal_list))
    num_images = len(pascal_list)
    a=np.load(pascal_list[0]).ravel()
    print(a.shape)
    num_features =a.shape[0]
    pascal_data = np.zeros((num_images,num_features))
    for i,pascal_file in enumerate(pascal_list):
        pascal_data[i,:] = np.load(pascal_file).ravel()
    taskonomy_data['pascal_voc_segmentation'] = pascal_data
    return taskonomy_data

def get_rdms(taskonomy_rdm_dir):
    #Load rdms
    dist_type = ['pearson', 'euclidean', 'cosine']
    tasks = ['autoencoder','class_1000', 'class_places', 'colorization','curvature',\
                       'denoise', 'edge2d', 'edge3d', \
                       'inpainting_whole','keypoint2d', 'keypoint3d', \
                       'reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','room_layout' ,\
                       'segment25d', 'segment2d', 'segmentsemantic', 'vanishing_point']
    print(len(tasks))
    rdms={}
    for dist in tqdm(dist_type):
        rdms[dist] = {}
        for task in tasks:
            rdm_path = os.path.join(taskonomy_rdm_dir,dist,task+"_encoder_output.mat.npy")
            #print(rdm_path)
            rdm = np.load(rdm_path)
            rdms[dist][task] = rdm
    return rdms
def rdm(activations_value,dist):
    if dist == 'pearson':
        RDM = 1-np.corrcoef(activations_value)
    elif dist == 'euclidean':
        RDM = euclidean_distances(activations_value)
    elif dist == 'cosine':
        RDM = 1- cosine_similarity(activations_value)
    return RDM

def get_similarity(x,y,kernel,similarity,debiased,centered,num_images,feature_ablation):
    if feature_ablation == 'None':
        x = x
        y = y
    elif feature_ablation == 'centering':
        x = (x - np.mean(x,axis=0))
        y = (y - np.mean(y,axis=0))
    elif feature_ablation == 'znorm':
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)
    elif feature_ablation == 'z_norm_my_implementation':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = z_norm_my_implementation(x)
        y = z_norm_my_implementation(y)

    elif feature_ablation == 'group_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = group_norm(x,group_size=2)
        y = group_norm(y,group_size=2)
    elif feature_ablation == 'instance_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = group_norm(x,group_size=8)
        y = group_norm(y,group_size=8)
    elif feature_ablation == 'layer_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = group_norm(x,group_size=1)
        y = group_norm(y,group_size=1)
    elif feature_ablation == 'batch_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = batch_norm(x)
        y = batch_norm(y)
    elif feature_ablation == 'covGL':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = covGL(x)
        y = covGL(y)
    elif feature_ablation == 'covLF':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = covLF(x)
        y = covLF(y)
    if similarity == 'cka':
        if kernel == 'linear':
            return cka(gram_linear(x),gram_linear(y),debiased=debiased,centered=centered)
        elif kernel == 'rbf':
            return cka(gram_rbf(x, 0.5),gram_rbf(y, 0.5),debiased=debiased,centered=centered)
        elif kernel == 'lap':
            return cka(gram_laplacian_scipy(x, 0.5),gram_laplacian_scipy(y, 0.5),debiased=debiased,centered=centered)
    elif similarity == 'spearman':
        if kernel == 'linear':
            return spearman(gram_linear(x),gram_linear(y),debiased=debiased,centered=centered,num_images=num_images)
        elif kernel == 'rbf':
            return spearman(gram_rbf(x, 0.5),gram_rbf(y, 0.5),debiased=debiased,centered=centered,num_images=num_images)
        elif kernel == 'lap':
            return spearman(gram_laplacian_scipy(x, 0.5),gram_laplacian_scipy(y, 0.5),debiased=debiased,centered=centered,num_images=num_images)
    elif similarity == 'pearson':
        if kernel == 'linear':
            return pearson(gram_linear(x),gram_linear(y),debiased=debiased,centered=centered,num_images=num_images)
        elif kernel == 'rbf':
            return pearson(gram_rbf(x, 0.5),gram_rbf(y, 0.5),debiased=debiased,centered=centered,num_images=num_images)
        elif kernel == 'lap':
            return pearson(gram_laplacian_scipy(x, 0.5),gram_laplacian_scipy(y, 0.5),debiased=debiased,centered=centered,num_images=num_images)

def get_rdms(taskonomy_rdm_dir):
    #Load rdms
    dist_type = ['pearson', 'euclidean', 'cosine']
    tasks = ['autoencoder','class_1000', 'class_places', 'colorization','curvature',\
                       'denoise', 'edge2d', 'edge3d', \
                       'inpainting_whole','keypoint2d', 'keypoint3d', \
                       'reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','room_layout' ,\
                       'segment25d', 'segment2d', 'segmentsemantic', 'vanishing_point']
    print(len(tasks))
    rdms={}
    for dist in tqdm(dist_type):
        rdms[dist] = {}
        for task in tasks:
            rdm_path = os.path.join(taskonomy_rdm_dir,dist,task+"_encoder_output.mat.npy")
            #print(rdm_path)
            rdm = np.load(rdm_path)
            rdms[dist][task] = rdm
    return rdms

def get_similarity_from_rdms(x,y,dist,similarity,debiased,centered,num_images,feature_ablation):
    if feature_ablation == 'None':
        x = x
        y = y
    elif feature_ablation == 'centering':
        x = (x - np.mean(x,axis = 0, keepdims=True))
        y = (y - np.mean(y,axis = 0, keepdims=True))
    elif feature_ablation == 'znorm':
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)
    elif feature_ablation == 'z_norm_my_implementation':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = z_norm_my_implementation(x)
        y = z_norm_my_implementation(y)
    elif feature_ablation == 'group_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = group_norm(x,group_size=2)
        y = group_norm(y,group_size=2)
    elif feature_ablation == 'instance_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = group_norm(x,group_size=8)
        y = group_norm(y,group_size=8)
    elif feature_ablation == 'layer_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = group_norm(x,group_size=1)
        y = group_norm(y,group_size=1)
    elif feature_ablation == 'batch_norm':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = batch_norm(x)
        y = batch_norm(y)
    elif feature_ablation == 'covGL':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = covGL(x)
        y = covGL(y)
    elif feature_ablation == 'covLF':
        x = np.reshape(x,(-1,16,16,8))
        y = np.reshape(y,(-1,16,16,8))
        x = covLF(x)
        y = covLF(y)
    if similarity == 'cka':
        return cka(rdm(x,dist),rdm(y,dist),debiased=debiased,centered=centered)
    elif similarity == 'spearman':
        return spearman(rdm(x,dist),rdm(y,dist),debiased=debiased,centered=centered,num_images=num_images)
    elif similarity == 'pearson':
        return pearson(rdm(x,dist),rdm(y,dist),debiased=debiased,centered=centered,num_images=num_images)



def main():
    parser = argparse.ArgumentParser(description='Comparison of similarity measures with transfer learning performance')
    parser.add_argument('-d','--dataset', help='dataset name', default = "nyuv2", type=str)
    args = vars(parser.parse_args())

    num_images = 200
    dataset = args['dataset']
    task_list = list_of_tasks.split(' ')
    print(task_list)
    taskonomy_feats_path = os.path.join('/home/kshitij/projects/CVPR_transferlearning/taskonomy_activations/',dataset)
    pascal_feats_path = '/home/kshitij/projects/taskonomy/deeplab_v3/CVPR2020_random_feat_pascal_' + dataset + '/'
    taskonomy_data = get_features(taskonomy_feats_path,pascal_feats_path)
                                    # Selecting random indices for a distribution
    num_images = 200 # for 2 random indices
    num_repetitions = 100

    num_total_images = 5000

    if dataset == 'nyuv2':
        num_total_images = 1449
    kernel_type = ['linear','lap','rbf']#,'rbf','lap']
    similarity_type = ['cka'] #'spearman',
    ablation_type = ['debiased_centered'] #'default',
    feature_ablation_type = ['None','znorm','group_norm','instance_norm','layer_norm','batch_norm']#'z_norm_my_implementation']#'covLF',
    save_dir = './CVPR2020_results/pascal_normcomparison_' + dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if True:
        save_path = os.path.join(save_dir,'kernels.npy')
        affinity_ablation = {}
        for kernel in (kernel_type):
            affinity_ablation[kernel]={}
            for similarity in similarity_type:
                affinity_ablation[kernel][similarity]={}
                for ablation in (ablation_type):
                    affinity_ablation[kernel][similarity][ablation] = {}
                    if ablation=='default':
                        debiased = False
                        centered = False
                    elif ablation=='centered':
                        debiased = False
                        centered = True
                    elif ablation=='debiased_centered':
                        debiased = True
                        centered = True
                    for feature_ablation in feature_ablation_type:
                        np.random.seed(1993)
                        indices = []
                        for i in range(num_repetitions):
                            indices.append(np.random.choice(range(num_total_images), num_images, replace=False))

                        #print(indices)
                        print(kernel,similarity,ablation,feature_ablation)
                        affinity_matrix = np.zeros((num_repetitions, len(task_list)), float)
                        for i in tqdm(range(num_repetitions)):
                            method = kernel + "__" + similarity +"__" + ablation +"__" + feature_ablation
                            start = time.time()
                            for index1,task1 in (enumerate(task_list)):
                                affinity_matrix[i,index1] = get_similarity(taskonomy_data[task1][indices[i],:],\
                                                                                taskonomy_data['pascal_voc_segmentation'][indices[i],:],\
                                                                                kernel,similarity,debiased,centered,num_images,feature_ablation)
                            end = time.time()
                            #print("Method is ", method)
                            #print("Time taken is ", end - start)
                        affinity_ablation[kernel][similarity][ablation][feature_ablation] = affinity_matrix

        np.save(save_path,affinity_ablation)


    save_path = os.path.join(save_dir,'rdms.npy')
    dist_type = ['pearson', 'euclidean', 'cosine']
    affinity_ablation = {}
    for dist in (dist_type):
        affinity_ablation[dist]={}
        for similarity in similarity_type:
            affinity_ablation[dist][similarity]={}
            for ablation in (ablation_type):
                affinity_ablation[dist][similarity][ablation] = {}

                if ablation=='default':
                    debiased = False
                    centered = False
                elif ablation=='centered':
                    debiased = False
                    centered = True
                elif ablation=='debiased_centered':
                    debiased = True
                    centered = True
                for feature_ablation in (feature_ablation_type):
                    np.random.seed(1993)
                    indices = []
                    for i in range(num_repetitions):
                        indices.append(np.random.choice(range(num_total_images), num_images, replace=False))
                    #print(indices)
                    print(dist,similarity,ablation,feature_ablation)
                    affinity_matrix = np.zeros((num_repetitions, len(task_list)), float)
                    for i in tqdm(range(num_repetitions)):
                        method = dist + "__" + similarity +"__" + ablation +"__" + feature_ablation
                        start = time.time()
                        for index1,task1 in (enumerate(task_list)):
                            affinity_matrix[i,index1] = get_similarity_from_rdms(taskonomy_data[task1][indices[i],:],\
                                                                        taskonomy_data['pascal_voc_segmentation'][indices[i],:],\
                                                                            dist,similarity,debiased,centered,num_images,feature_ablation)
                            #print(affinity_matrix[i,index1])
                        end = time.time()
                        #print("Method is ", method)
                        #print("Time taken is ", end - start)
                    affinity_ablation[dist][similarity][ablation][feature_ablation]=affinity_matrix
    np.save(save_path,affinity_ablation)

if __name__ == "__main__":
    main()
