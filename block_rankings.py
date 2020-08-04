import seaborn as sns
import numpy as np
import scipy.io as sio
import os
import glob
from tqdm import tqdm
import numpy as np
from utils import *
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from sklearn.preprocessing import StandardScaler


def get_features(pascal_feats_path,taskonomy_feats_path,block):
    pascal_list = glob.glob(pascal_feats_path+"*"+block+".npy")
    pascal_list.sort()
    num_images_2use = 200
    pascal_list=pascal_list[:num_images_2use]
    print(len(pascal_list))
    pascal_image_list = []
    for image in pascal_list:
        pascal_image_list.append(image.split('.')[0].split("/")[-1])
    taskonomy_tasks = ['autoencoder','class_1000', 'class_places', 'colorization','curvature',\
                       'denoise', 'edge2d', 'edge3d', \
                       'inpainting_whole','keypoint2d', 'keypoint3d', \
                       'reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','room_layout' ,\
                       'segment25d', 'segment2d', 'segmentsemantic', 'vanishing_point']
    print(len(taskonomy_tasks))
    taskonomy_list={}
    for task in taskonomy_tasks:
        taskonomy_list[task] = glob.glob(taskonomy_feats_path+"*"+  task +"_encoder_output.npy")
        taskonomy_list[task].sort()
        taskonomy_list[task] = taskonomy_list[task][:num_images_2use]
        print(task, len(taskonomy_list[task]))

    #Loading data
    num_images = len(pascal_list)
    a=np.load(pascal_list[0]).ravel()
    print(a.shape)
    num_features =a.shape[0]
    pascal_data = np.zeros((num_images,num_features))
    for i,pascal_file in enumerate(pascal_list):
        pascal_data[i,:] = np.load(pascal_file).ravel()
    taskonomy_data = {}
    for task in taskonomy_tasks:
        a=np.load(taskonomy_list[task][0]).ravel()
        print(a.shape)
        num_features =a.shape[0]
        taskonomy_data[task] = np.zeros((num_images,num_features))
        for i,taskonomy_file in enumerate(taskonomy_list[task]):
            taskonomy_data[task][i,:]  = np.load(taskonomy_file).ravel()
    return pascal_data,taskonomy_data


tasks = ['autoencoder','class_1000', 'class_places', 'colorization','curvature',\
                   'denoise', 'edge2d', 'edge3d', \
                   'inpainting_whole','keypoint2d', 'keypoint3d', \
                   'reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','room_layout' ,\
                   'segment25d', 'segment2d', 'segmentsemantic', 'vanishing_point']
tasks_2D =['autoencoder', 'colorization',\
                   'denoise', 'edge2d',  \
                   'inpainting_whole','keypoint2d', \
                    'segment2d']
tasks_3D = ['curvature', 'edge3d','reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','segment25d']
tasks_semantic = ['class_1000', 'class_places','segmentsemantic']

def main():

    parser = argparse.ArgumentParser(description='Comparison of similarity measures with transfer learning performance')
    parser.add_argument('-d','--dataset', help='dataset name', default = "features", type=str)
    args = vars(parser.parse_args())
    dataset = args['dataset']

    imagenet_feats_path = '/home/kshitij/projects/CVPR_transferlearning/imagenetr50_activations/' + dataset + '/'
    if dataset=='features':
        taskonomy_feats_path = '/home/kshitij/projects/CVPR_transferlearning/taskonomy_activations/taskonomy_5000/'
    else:
        taskonomy_feats_path = '/home/kshitij/projects/CVPR_transferlearning/taskonomy_activations/' + dataset +'/'

    save_dir = './CVPR2020_results'
    if not os.path.exists(os.path.join(save_dir,dataset)):
        os.makedirs(os.path.join(save_dir,dataset))
    save_path = os.path.join(save_dir,dataset,'rankings_')

    blocks = ['block_1','block_2','block_3','block_4']
    rankings = {}
    for block in blocks:
        imagenet_data,taskonomy_data = get_features(imagenet_feats_path,taskonomy_feats_path,block)
        similarity_types = ['rbf_cka','cka','cca']
        cka_task = {}

        rbf_cka = {}
        linear_cka = {}
        cca_dict = {}
        imagenet_data = StandardScaler().fit_transform(imagenet_data)
        for task in tqdm(tasks):
            #print(corr_samples[i].keys())
            taskonomy_data[task] = StandardScaler().fit_transform(taskonomy_data[task])
            rbf_cka[task] = cka(gram_laplacian_scipy(imagenet_data, 0.5)\
                                     ,gram_laplacian_scipy(taskonomy_data[task], 0.5),debiased=True)
            #linear_cka[task] = cka(gram_linear(imagenet_data)\
            #                           , gram_linear(taskonomy_data[task]))
            #cca_dict[task] = cca(imagenet_data, taskonomy_data[task])

        rankings[block] = rbf_cka
        np.save(save_path+block,rbf_cka)
        plt.figure()
        x=[]
        y=[]
        for w in sorted(rbf_cka, key=rbf_cka.get, reverse=True):
            x.append(w)
            y.append(rbf_cka[w])
            print (w, rbf_cka[w])

        barlist = plt.bar(range(len(rbf_cka)), list(y), align='center')
        for i in range(len(x)):
            if x[i] in tasks_2D:
                barlist[i].set_color('b')
            elif x[i] in tasks_3D:
                barlist[i].set_color('g')

            elif x[i] in tasks_semantic:
                barlist[i].set_color('m')
            else:
                barlist[i].set_color('r')


        plt.xticks(range(len(rbf_cka)), list(x))
        plt.xticks(rotation=70)
        plt.savefig(save_path+block + '.png', bbox_inches="tight")
        plt.savefig(save_path+block + '.pdf', bbox_inches="tight")
        plt.savefig(save_path+block + '.eps', bbox_inches="tight")

if __name__ == "__main__":
    main()
