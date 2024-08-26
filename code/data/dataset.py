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
                                                                                                      
        self.instance_names = sorted(os.listdir(self.dataset_root))
        examples = []

        for idx, instance_name in enumerate(self.instance_names):

            if not train and idx > 239:
                break

            if idx > 551:
                break

            print("Reading instance idx: {}".format(idx))
            instance = self._read_instance(instance_name)
            text = instance['text']                                                     
            kps = instance['kps']
            s_feature = instance["sent_embeddings"]
            z = instance["z"]
            examples.append(data.Example.fromlist([text, kps, s_feature, 0, 0, z, idx, instance_name.replace(".pkl", "")], fields))
                                                                                                      
        super(SignProdDataset, self).__init__(examples, fields, **kwargs)

    def _read_instance(self, instance_name):
        instance_fp = os.path.join(self.dataset_root, instance_name)
        with open(instance_fp, "rb") as handler:
            instance = pickle.load(handler)
        return instance









    
