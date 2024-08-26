import os
import pickle
import torch
import numpy as np
from PIL import Image
from torchtext import data
from torchvision import datasets, models, transforms

MAX_SENTENCE_SIZE = 38
PAD_IDX = 0

def pad_pos(pos):
    diff = MAX_SENTENCE_SIZE - len(pos)
    for _ in range(0, diff):
        pos.append(PAD_IDX)
    return pos
    

class SignProdDataset(data.Dataset):                                                                                                                                                                        
                                                                                                                                                                                                            
    def __init__(self, dataset_root, train, fields, **kwargs):                                                   

        self.dataset_root = dataset_root                                                                                                                                                                                          

        if self.dataset_root is None: 
            print("Please input a data root")
            exit(0)

        if train:
            print("Reading train")                                                                                      
            self.instance_names = sorted(os.listdir(self.dataset_root))[:5870]
            idx = 0
        else:
            print("Reading validation")
            self.instance_names = sorted(os.listdir(self.dataset_root))[5870:6070]
            idx = 5870

        examples = []

        for instance_name in self.instance_names:

            if idx > 6070:
                break

            print("Reading instance idx: {}".format(idx))
            instance = self._read_instance(instance_name)
            text = instance['text']                                                     
            kps = instance['kps']
            pos = instance["pos"]
            sent_feature = instance["sent_embeddings"]
            sem_feature = instance["sem_embeddings"]
            z = instance["z"]
            aus = instance["aus"]
            pos = pad_pos(pos)
            examples.append(data.Example.fromlist([text, kps, sent_feature,sem_feature, pos, aus, z, idx, instance_name.replace(".pkl", "")], fields))
            idx += 1
                                                                                                      
        super(SignProdDataset, self).__init__(examples, fields, **kwargs)

    def _read_instance(self, instance_name):
        instance_fp = os.path.join(self.dataset_root, instance_name)
        with open(instance_fp, "rb") as handler:
            instance = pickle.load(handler)
        return instance









    
