# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:17:00 2020

@author: kshitij
"""

import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import time
from utils import  get_similarity,get_similarity_from_rdms


list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole pascal_voc_segmentation'


def get_features(taskonomy_feats_path,pascal_feats_path,features_filename):
    """
    Parameters
    ----------
    taskonomy_feats_path : string
        path to directory containing features from taskonomy models
    pascal_feats_path : string
        path to directory containing features from pascal voc model

    Returns
    -------
    taskonomy_data : dict
        dictionary containg features of taskonomy models and pascal voc model.

    """
    if os.path.isfile(features_filename):
        start = time.time()
        taskonomy_data = np.load(features_filename,allow_pickle=True)
        end = time.time()
        print("whole file loading time is ", end - start)
        return taskonomy_data.item()

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
    for task in tqdm(taskonomy_tasks):
        taskonomy_data[task] = np.zeros((num_images,num_features))
        for i,taskonomy_file in tqdm(enumerate(taskonomy_list[task])):
            taskonomy_data[task][i,:]  = np.load(taskonomy_file).ravel()

    pascal_list = glob.glob(pascal_feats_path+"/*.npy")
    pascal_list.sort()
    print(len(pascal_list))
    num_images = len(pascal_list)
    a=np.load(pascal_list[0]).ravel()
    print(a.shape)
    num_features =a.shape[0]
    pascal_data = np.zeros((num_images,num_features))
    for i,pascal_file in tqdm(enumerate(pascal_list)):
        pascal_data[i,:] = np.load(pascal_file).ravel()
    taskonomy_data['pascal_voc_segmentation'] = pascal_data
    np.save(features_filename, taskonomy_data)
    return taskonomy_data




def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Taskonomy Tasks')
    parser.add_argument('-d','--dataset', help='image dataset to use for computing DDS: options are [pascal_5000, taskonomy_5000, nyuv2]', default = "pascal_5000", type=str)
    parser.add_argument('-fd','--feature_dir', help='path to saved features root directory', default = "./features/", type=str)
    parser.add_argument('-fdt','--feature_dir_taskonomy', help='path to saved features from taskonomy models', default = "./features/taskonomy_activations/", type=str)
    parser.add_argument('-fdp','--feature_dir_pascal', help='path to saved features from pascal models', default = "./features/pascal_activations/", type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', default = "./results/DDScomparison_pascal", type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 200, type=int)
    parser.add_argument('-i','--num_iters', help='number of iterations for bootstrap', default = 100, type=int)
    args = vars(parser.parse_args())

    num_images = args['num_images']
    dataset = args['dataset']
    taskonomy_feats_path = os.path.join(args['feature_dir_taskonomy'],dataset)
    pascal_feats_path = os.path.join(args['feature_dir_pascal'],dataset)
    num_repetitions = args['num_iters']
    features_filename = os.path.join("./features","taskonomy_pascal_feats_" + args['dataset'] + ".npy")
    num_total_images = 5000
    if dataset == 'nyuv2':
        num_total_images = 1449
    save_dir = os.path.join(args['save_dir'],dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    task_list = list_of_tasks.split(' ')
    print(task_list)

    taskonomy_data = get_features(taskonomy_feats_path,pascal_feats_path,features_filename)


    # setting up DDS using Q,D,f,g for kernels
    kernel_type = ['rbf','lap','linear'] # possible kernels (f in DDS)
    feature_norm_type = ['znorm'] # possible normalizations (Q,D in DDS)

    save_path = os.path.join(save_dir,'kernels.npy')
    affinity_ablation = {}
    for kernel in (kernel_type):
        affinity_ablation[kernel]={}
        for feature_norm in feature_norm_type:
            np.random.seed(1993)
            indices = []
            for i in range(num_repetitions):
                indices.append(np.random.choice(range(num_total_images), num_images, replace=False))
            print(kernel,feature_norm)
            affinity_matrix = np.zeros((num_repetitions, len(task_list)), float)
            for i in tqdm(range(num_repetitions)):
                method = kernel  +"__" + feature_norm
                start = time.time()
                for index1,task1 in (enumerate(task_list)):
                    affinity_matrix[i,index1] = get_similarity(taskonomy_data[task1][indices[i],:],\
                                                               taskonomy_data['pascal_voc_segmentation'][indices[i],:],\
                                                               kernel,feature_norm)
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
        for feature_norm in feature_norm_type:
            np.random.seed(1993)
            indices = []
            for i in range(num_repetitions):
                indices.append(np.random.choice(range(num_total_images), num_images, replace=False))
            print(dist,feature_norm)
            affinity_matrix = np.zeros((num_repetitions, len(task_list)), float)
            for i in tqdm(range(num_repetitions)):
                method = dist  +"__" + feature_norm
                start = time.time()
                for index1,task1 in (enumerate(task_list)):
                    affinity_matrix[i,index1] = get_similarity_from_rdms(taskonomy_data[task1][indices[i],:],\
                                                               taskonomy_data['pascal_voc_segmentation'][indices[i],:],\
                                                               dist,feature_norm)
                end = time.time()
                print("Method is ", method)
                print("Time taken is ", end - start)
            affinity_ablation[dist][feature_norm] = affinity_matrix

    np.save(save_path,affinity_ablation)


if __name__ == "__main__":
    main()
