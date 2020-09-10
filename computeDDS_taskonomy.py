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



def get_features(features_filename,num_images):
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
    task_list = list_of_tasks.split(' ')
    if os.path.isfile(features_filename):
        start = time.time()
        taskonomy_data = np.load(features_filename,allow_pickle=True)
        end = time.time()
        print("whole file loading time is ", end - start)
        taskonomy_data_full = taskonomy_data.item()
        taskonomy_data_few_images = {}
        for index,task in enumerate(task_list):
            taskonomy_data_few_images[task] = taskonomy_data_full[task][:num_images,:]
        return taskonomy_data_few_images

def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Taskonomy Tasks')
    parser.add_argument('-d','--dataset', help='image dataset to use for computing DDS: options are [pascal_5000, taskonomy_5000, nyuv2]', default = "taskonomy_5000", type=str)
    parser.add_argument('-fd','--feature_dir', help='path to saved features from taskonomy models', default = "./features/", type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', default = "./results/DDScomparison_taskonomy", type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 200, type=int)
    args = vars(parser.parse_args())

    num_images = args['num_images']
    dataset = args['dataset']
    features_filename = os.path.join(args['feature_dir'],"taskonomy_pascal_feats_" + args['dataset'] + ".npy")
    save_dir = os.path.join(args['save_dir'],dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    task_list = list_of_tasks.split(' ')

    taskonomy_data = get_features(features_filename,num_images) # function that returns features from taskonomy models for first #num_images


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
