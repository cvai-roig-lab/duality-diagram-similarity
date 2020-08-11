# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:30:34 2020
Compute DDS between Taskonomy tasks
@author: kshitij
"""

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import time

from utils import  get_similarity_from_rdms,get_similarity



def get_features(taskonomy_feats_path,num_images):
    """

    Parameters
    ----------
    taskonomy_feats_path : TYPE
        DESCRIPTION.
    num_images : int 
        number of images to compute DDS

    Returns
    -------
    taskonomy_data : dict
        dictionary containg features of taskonomy models.

    """
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
        taskonomy_list[task] = taskonomy_list[task][:num_images]
        print(task, len(taskonomy_list[task]))

    #Loading data
    #num_images = len(taskonomy_list[task])
    print(np.load(taskonomy_list[task][0]).shape)
    a=np.load(taskonomy_list[task][0]).ravel()
    print(a.shape)
    num_features =a.shape[0]
    taskonomy_data = {}
    for task in taskonomy_tasks:
        taskonomy_data[task] = np.zeros((num_images,num_features))
        for i,taskonomy_file in enumerate(taskonomy_list[task]):
            taskonomy_data[task][i,:]  = np.load(taskonomy_file).ravel()
    return taskonomy_data

def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Taskonomy Tasks')
    parser.add_argument('-d','--dataset', help='image dataset to use for computing DDS', default = "nyuv2", type=str)
    parser.add_argument('-fd','--feature_dir', help='path to saved features from taskonomy models', default = "./features/taskonomy_activations/", type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', default = "./results/DDScomparison_taskonomy", type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 200, type=int)
    args = vars(parser.parse_args())

    num_images = args['num_images']
    dataset = args['dataset']
    taskonomy_feats_path = os.path.join(args['feature_dir'],dataset)
    save_dir = os.path.join(args['save_dir'],dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    task_list = list_of_tasks.split(' ')
    
    taskonomy_data = get_features(taskonomy_feats_path,num_images) # function that returns features from taskonomy models for first #num_images 
    
    
    # setting up DDS using Q,D,f,g for kernels
    kernel_type = ['rbf','lap','linear'] # possible kernels (f in DDS)
    feature_norm_type = ['None','centering','znorm','group_norm','instance_norm','layer_norm','batch_norm'] # possible normalizations (Q,D in DDS)
    
    
    save_path = os.path.join(save_dir,'kernels.npy')
    affinity_ablation = {}
    for kernel in (kernel_type):
        affinity_ablation[kernel]={}
        for feature_norm in feature_norm_type:
            affinity_matrix = np.zeros((len(task_list), len(task_list)), float)
            method = kernel + "__" + feature_norm
            start = time.time()
            for index1,task1 in tqdm(enumerate(task_list)):
                for index2,task2 in (enumerate(task_list)):
                    if index1 > index2:
                        continue
                    affinity_matrix[index1,index2] = get_similarity(taskonomy_data[task1],\
                                                                    taskonomy_data[task2],\
                                                                    kernel,feature_norm)
                    affinity_matrix[index2,index1] = affinity_matrix[index1,index2]
            end = time.time()
            print("Method is ", method)
            print("Time taken is ", end - start)
            affinity_ablation[kernel][feature_norm] = affinity_matrix

        np.save(save_path,affinity_ablation)

    # setting up DDS using Q,D,f,g for distance functions
    save_path = os.path.join(save_dir,'rdms.npy')
    dist_type = ['pearson', 'euclidean', 'cosine']
    affinity_ablation = {}
    for dist in (dist_type):
        affinity_ablation[dist]={}
        for feature_norm in (feature_norm_type):
            affinity_matrix = np.zeros((len(task_list), len(task_list)), float)
            method = dist + "__" + feature_norm
            start = time.time()
            for index1,task1 in tqdm(enumerate(task_list)):
                for index2,task2 in enumerate(task_list):
                    if index1 > index2:
                        continue
                    affinity_matrix[index1,index2] = get_similarity_from_rdms(taskonomy_data[task1],\
                                                                              taskonomy_data[task2],\
                                                                              dist,feature_norm)
                    affinity_matrix[index2,index1] = affinity_matrix[index1,index2]
            end = time.time()
            print("Method is ", method)
            print("Time taken is ", end - start)
            affinity_ablation[dist][feature_norm]=affinity_matrix
    np.save(save_path,affinity_ablation)

if __name__ == "__main__":
    main()

