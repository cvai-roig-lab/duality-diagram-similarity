import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import chi2_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,rbf_kernel
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import numpy as np
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

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'

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

def get_features(taskonomy_feats_path):
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
    a=np.load(taskonomy_list[task][0]).ravel()
    print(a.shape)
    num_features =a.shape[0]
    taskonomy_data = {}
    for task in taskonomy_tasks:
        taskonomy_data[task] = np.zeros((num_images,num_features))
        for i,taskonomy_file in enumerate(taskonomy_list[task]):
            taskonomy_data[task][i,:]  = np.load(taskonomy_file).ravel()
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
        x = (x - np.mean(x,axis=0))
        y = (y - np.mean(y,axis=0))
    elif feature_ablation == 'znorm':
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)
    if similarity == 'cka':
        return cka(rdm(x,dist),rdm(y,dist),debiased=debiased,centered=centered)
    elif similarity == 'spearman':
        return spearman(rdm(x,dist),rdm(y,dist),debiased=debiased,centered=centered,num_images=num_images)




def main():
    parser = argparse.ArgumentParser(description='Comparison of similarity measures with transfer learning performance')
    parser.add_argument('-d','--dataset', help='dataset name', default = "nyuv2", type=str)
    args = vars(parser.parse_args())

    num_images = 200
    dataset = args['dataset']
    task_list = list_of_tasks.split(' ')
    print(task_list)
    taskonomy_feats_path = os.path.join('/home/kshitij/projects/CVPR_transferlearning/taskonomy_activations/',dataset)
    taskonomy_data = get_features(taskonomy_feats_path)
                                    # Selecting random indices for a distribution
    num_images = 200 # for 2 random indices
    num_repetitions = 1
    num_total_images = 5000

    kernel_type = ['linear','rbf','lap']
    similarity_type = ['spearman','cka']
    ablation_type = ['default','debiased_centered']
    feature_ablation_type = ['None','znorm']
    save_dir = './CVPR2020_results/ablation_' + dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if True:
        save_path = os.path.join(save_dir,'kernels.npy')
        affinity_ablation = {}
        for kernel in tqdm(kernel_type):
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
                        indices.append(range(num_images))
                        print(indices)
                        affinity_matrix = np.zeros((num_repetitions,len(task_list), len(task_list)), float)
                        for i in range(num_repetitions):
                            method = kernel + "__" + similarity +"__" + ablation +"__" + feature_ablation
                            start = time.time()
                            for index1,task1 in tqdm(enumerate(task_list)):
                                for index2,task2 in enumerate(task_list):
                                    #print(corr_samples[i].keys())
                                    if index1 > index2:
                                        continue
                                    affinity_matrix[i,index1,index2] = get_similarity(taskonomy_data[task1][indices[i],:],\
                                                                                    taskonomy_data[task2][indices[i],:],\
                                                                                    kernel,similarity,debiased,centered,num_images,feature_ablation)
                                    affinity_matrix[i,index2,index1] = affinity_matrix[i,index1,index2]
                            end = time.time()
                            print("Method is ", method)
                            print("Time taken is ", end - start)
                        affinity_ablation[kernel][similarity][ablation][feature_ablation] = affinity_matrix
        #np.save(save_path,affinity_ablation)


    save_path = os.path.join(save_dir,'rdms.npy')
    dist_type = ['pearson', 'euclidean', 'cosine']
    affinity_ablation = {}
    for dist in tqdm(dist_type):
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
                for feature_ablation in feature_ablation_type:
                    indices = []
                    indices.append(range(num_images))
                    print(indices)
                    affinity_matrix = np.zeros((num_repetitions,len(task_list), len(task_list)), float)
                    for i in range(num_repetitions):
                        method = dist + "__" + similarity +"__" + ablation +"__" + feature_ablation
                        start = time.time()
                        for index1,task1 in tqdm(enumerate(task_list)):
                            for index2,task2 in enumerate(task_list):
                                #print(corr_samples[i].keys())
                                if index1 > index2:
                                    continue
                                else:
                                    affinity_matrix[i,index1,index2] = get_similarity_from_rdms(taskonomy_data[task1][indices[i],:],\
                                                                                    taskonomy_data[task2][indices[i],:],\
                                                                                    dist,similarity,debiased,centered,num_images,feature_ablation)
                                    affinity_matrix[i,index2,index1] = affinity_matrix[i,index1,index2]
                        end = time.time()
                        print("Method is ", method)
                        print("Time taken is ", end - start)
                    affinity_ablation[dist][similarity][ablation][feature_ablation]=affinity_matrix
    #np.save(save_path,affinity_ablation)

if __name__ == "__main__":
    main()
