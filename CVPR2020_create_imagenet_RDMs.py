##Creates RDM for 50, 100, 200 image dataset from taskonomy testset

import json
import glob
import os
import numpy as np
import datetime
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm

# PCA related settings
pca = PCA(n_components=45)
scaler = StandardScaler()

#RDM from activations
def create_rdm(task_list, RDM_dir,save_dir,use_znorm,use_pca,use_corr):
    fc_task_list = 'class_1000 class_places vanishing_point jigsaw room_layout vanishing_point'
    fc_task_list = fc_task_list.split()
    feedforward_encoder_save_list = 'feedforward_encoder_block4 encoder_output prefinal'
    feedforward_encoder_save_list = feedforward_encoder_save_list.split()
    for task in task_list:
        print(task)
        activations = glob.glob(RDM_dir + "/*" +task+ ".npy")
        activations.sort()
        print(activations)
        n = len(activations)
        RDM = np.zeros((n,n))
        RDM_filename = task +".mat"
        if use_znorm:
            feature_0=np.load(activations[0])
            print(feature_0.shape,n)
            reshape_dim=1
            if len(feature_0.shape)>2:
                for l in range(len(feature_0.shape)-1):
                    reshape_dim*=feature_0.shape[l+1]
                activations_value = np.zeros((n,reshape_dim))
            else:
                activations_value = np.zeros((n,feature_0.shape[1]))
            for i in range(n):
                activation = np.load(activations[i])[0,:]
                #print(activation.shape)
                if len(feature_0.shape)>2:
                    activation_reshaped = np.reshape(activation,(reshape_dim,))
                else:
                    activation_reshaped = activation
                activations_value[i,:]= activation_reshaped
            #scaler = StandardScaler()
            # Fit on training set only.
            #scaler.fit(activations_value)
            # Apply transform to both the training set and the test set.
            #activations_value = scaler.transform(activations_value)
            activations_value = (activations_value - np.mean(activations_value,axis=0))#/(np.sqrt(np.var(activations_value,axis=0))+1e-10)
            #print(np.mean(activations_value,axis=0).shape)
            if use_pca:
                pca.fit(activations_value)
                activations_value = pca.transform(activations_value)
                activations_value = (activations_value - np.mean(activations_value))/(np.sqrt(np.var(activations_value)+1e-10))
                print(activations_value.shape,pca.n_components_)

            for i in tqdm(range(n)):
                #print(activations[i])
                for j in tqdm(range(n)):
                    if use_znorm:
                        feature_i=activations_value[i]
                        feature_j=activations_value[j]
                        #print(feature_i.shape)
                    else:
                        feature_i = np.load(activations[i])
                        feature_j = np.load(activations[j])
                        #print(feature_i.shape)
                        if len(feature_i.shape)>2:
                            reshape_dim=1
                            for l in range(len(feature_i.shape)-1):
                                reshape_dim*=feature_i.shape[l+1]

                            feature_i = np.reshape(feature_i,(reshape_dim))
                            feature_j = np.reshape(feature_j,(reshape_dim))

                    if use_corr:
                        RDM[i,j] = 1-np.corrcoef(feature_i,feature_j)[0][1]
                    else:
                        RDM[i,j] = np.mean((feature_i-feature_j)**2)
            sio.savemat(os.path.join(save_dir,RDM_filename),dict(rdm= RDM))
            print(os.path.join(save_dir,RDM_filename),RDM.shape)


def main():
    parser = argparse.ArgumentParser(description='Creates RDM from DNN activations')
    parser.add_argument('-rd','--RDM_dir', help='RDM directory path', default = "/home/kshitij/projects/taskonomy/taskonomy_rdm/taskbank/tools/resnet_imagenet_feats_taskonomy500", type=str)
    parser.add_argument('-sd','--save_dir', help='save directory path', default = "/home/kshitij/projects/taskonomy/taskonomy_rdm/taskbank/tools/resnet_imagenet_rdms", type=str)
    parser.add_argument('-z','--z_norm', help='To use znorm or not', default = True, type=int)
    parser.add_argument('-pca','--pca', help='To use PCA or not', default=False,type=int)
    parser.add_argument('-d','--dissimilarity_function', help='To use 1-correlation or simple euclidean', default=True,type=int)
    args = vars(parser.parse_args())
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

    args_file = os.path.join(args['save_dir'],'args.json')
    with open(args_file, 'w') as fp:
        json.dump(args, fp, sort_keys=True, indent=4)

    use_znorm = args['z_norm']
    use_pca = args['pca']
    use_corr = args['dissimilarity_function']
    print(use_znorm,use_pca,use_corr)

    #segmentation_ RDMS
    #task_list = 'class_1000 autoencoder curvature denoise edge2d edge3d \
    #keypoint2d keypoint3d colorization \
    #reshade rgb2depth rgb2mist rgb2sfnorm \
    #room_layout segment25d segment2d vanishing_point \
    #segmentsemantic class_1000 class_places inpainting_whole'
    task_list = 'block_3 block_2 block_3 block_4'
    task_list = task_list.split()
    # tasks with fc layers in decoder
    #list_of_tasks = ['class_places']

    create_rdm(task_list, args['RDM_dir'],args['save_dir'],use_znorm,use_pca,use_corr)








if __name__ == "__main__":
    main()
