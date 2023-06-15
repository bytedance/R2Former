# This code is only for reference, you may need to modify it to run on your machine.
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
from PIL import Image
import torch.nn.functional as F
from functools import partial
import time
import cupy as cp
from numpy import linalg as LA
import datasets_ws
from model.network import GeoLocalizationNet, GeoLocalizationNetRerank
from torch.utils.data.dataloader import DataLoader
import util
from tqdm import tqdm
import test
import random
from ptflops import get_model_complexity_info

base_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def scan_training(args):
    # args.resume = 'logs/deits_msls/2022-09-04_12-52-51/best_model.pth'
    # args.resume = 'logs/deits_msls_v2/2022-09-27_14-25-33/best_model.pth'
    #===================================================================
    # args.backbone = 'deitBase'
    # args.resume = 'msls_v2_deitBase.pth'
    #===================================================================
    args.backbone = 'resnet50'
    args.resume = 'msls_v2_resnet50gem.pth'
    #===================================================================
    triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder
                                              , args.dataset_name, "train",
                                              args.negs_num_per_query)
    triplets_ds.is_inference = True
    subset_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                           batch_size=args.infer_batch_size, shuffle=False,
                           pin_memory=(args.device == "cuda"))
    model = GeoLocalizationNetRerank(args)
    model = torch.nn.DataParallel(model)
    model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    model = model.eval().cuda()
    model.module.single = True

    cache = np.zeros([len(triplets_ds), args.fc_output_dim], dtype=np.float32)
    print(cache.shape, triplets_ds.queries_num, triplets_ds.database_num)
    with torch.no_grad():
        for images, indexes in tqdm(subset_dl, ncols=100):
            images = images.to(args.device)
            features = model(images)
            cache[indexes.numpy()] = features.cpu().numpy()
    np.save('result/'+args.dataset_name+'_v2_'+args.backbone+'_reference_feature.npy', cache[:triplets_ds.database_num])
    np.save('result/'+args.dataset_name+'_v2_'+args.backbone+'_query_feature.npy', cache[triplets_ds.database_num:])


def compute_mining(args):
    args.backbone = 'deitBase' #'resnet50' #
    query_feature = cp.load('result/'+args.dataset_name+'_v2_'+args.backbone+'_query_feature.npy')
    reference_features = cp.load('result/'+args.dataset_name+'_v2_'+args.backbone+'_reference_feature.npy')
    # features = cp.load('result/train_features_GLD.npy')
    N_q, C = query_feature.shape
    N_r, C = reference_features.shape

    loader = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train",
                                              args.negs_num_per_query)
    # ================================================
    interval = 120
    keep = 500
    hard_init = np.zeros([N_q, keep], dtype=int)
    # ==================================================
    for i in range(0, N_q, interval):
        t_s = time.time()
        similarity = cp.matmul(query_feature[i:(i + interval)], reference_features.transpose())
        order = cp.argsort(-similarity, axis=1).astype(int)
        hard_init[i:(i + interval)] = order[:, :keep].get()
        # print(hard_init[:20])
        # raise Exception
        print(i, 'time:', time.time()-t_s, 'estimated:', (N_q-i)/interval*(time.time()-t_s)/3600)
    np.save('result/' + args.dataset_name + '_v2_' + args.backbone + '_hard_init.npy', hard_init)
    # # ====================================================
    interval = 120
    keep = 200
    hard_init = cp.load('result/' + args.dataset_name + '_v2_' + args.backbone + '_hard_init.npy')
    hard_final = np.zeros([N_q, keep], dtype=int) - 1
    # ==================================================
    count = 0
    for i in range(N_q):
        t_s = time.time()
        positive = cp.asarray(loader.soft_positives_per_query[i])
        temp = cp.concatenate([positive, hard_init[i]])
        _, idx = np.unique(temp, return_index=True)
        temp = temp[np.sort(idx)][len(positive):]
        if len(temp) < keep:
            hard_final[i, :len(temp)] = temp.get()
            count += 1
            print(i, len(temp), len(positive), count)
        else:
            hard_final[i] = temp[:keep].get()
        if i % 100 == 0:
            print(i, 'time:', time.time() - t_s, 'estimated:', (N_q - i) * (time.time() - t_s) / 3600)
        # raise Exception
    np.save('result/' + args.dataset_name + '_v2_' + args.backbone + '_hard_final_distance.npy', hard_final)

class Args():
    def __init__(self):
        self.num_workers = 8
        self.resize = [480, 640]
        self.backbone = 'deit'
        self.aggregation = 'gem'
        self.datasets_folder = '../datasets_vg/datasets'
        self.negs_num_per_query = 1
        self.dataset_name = 'msls'
        self.infer_batch_size = 256
        self.resume = ''
        self.test_method = 'hard_resize'
        self.val_positive_dist_threshold = 25 #25
        self.train_positives_dist_threshold = 10
        self.mining = 'full'
        self.neg_samples_num = 1000
        self.brightness = None
        self.contrast = None
        self.saturation = None
        self.hue = None
        self.rand_perspective = None
        self.horizontal_flip = False
        self.random_resized_crop = None
        self.random_rotation = None
        self.device = 'cuda'
        self.local_dim = 128
        self.fc_output_dim = 256
        self.features_dim = 256
        self.hypercolumn = 0
        self.rerank_batch_size = 8
        self.efficient_ram_testing = False
        self.recall_values = [1, 5, 10, 20, 100]
        self.rerank_model = 'correlation_transformer'
        self.num_local = 500
        self.pretrain = 'imagenet'
        self.l2 = 'before_pool'
        self.non_local = False
        self.finetune = 0

# This code is only for reference, you may need to modify it to run on your machine.
if __name__ == '__main__':
    args = Args()
    # scan_training(args)
    # compute_mining(args)
