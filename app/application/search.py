import os
import pathlib
import pickle

import numpy as np
import torch
import tqdm
import faiss
import open_clip



def eval(files, index, features, k=2048):
    features = features.astype(np.float32)
    faiss.normalize_L2(features)
    if k > len(files):
        k = len(files)
    
    index.nprobe = 64

    print(index.ntotal)
    D, I = index.search(features, k=k)
    
    temp = [0 for i in files]
    for i, d in zip(I, D):
        for item_id, distance in zip(i, d):
            temp[item_id] += distance
    scores = []
    for i, score in enumerate(temp):
        scores.append([files[i], score])
    return scores