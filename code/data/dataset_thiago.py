import os
import pickle
import torch
import numpy as np
from PIL import Image
from torchtext import data
from torchvision import datasets, models, transforms
from einops import rearrange
import io
from torch.utils.data import Dataset


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
            self.instance_names = sorted(os.listdir(self.dataset_root))[:5970]
            idx = 0
        else:
            self.instance_names = sorted(os.listdir(self.dataset_root))[5970:6070]
            idx = 5970

        examples = []

        for instance_name in self.instance_names:

            if idx > 6071:
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


def read_phoenix_data(data_path, skip_frames=1):

    print(f"Loading phoenix {data_path.split('/')[-1]} set")

    trg_size = 361
    samples = []    
    trg_path = f"{data_path}.skels"
    line_count = 0
    with io.open(trg_path, mode='r', encoding='utf-8') as trg_file:     

        i = 0

        for trg_line in trg_file:

            i+= 1
            print("Processing line: {}".format(line_count))
            trg_line = trg_line.strip()
            trg_line = trg_line.split(" ")
            if len(trg_line) == 1:
                continue

            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]

            if trg_line != '':
                trg_frames = np.array(trg_frames)[..., :-1]
                trg_frames = rearrange(trg_frames, 't (v c) -> t v c', v=120, c=3)[:, :, :2]
                samples.append(trg_frames)
                line_count += 1

    return samples

def chunkenize(
    input_joints,
    seq_len
):

    assert type(input_joints) == list, 'Inputs must be list of tensors.'

    z_list = []
    for joint_feat in input_joints:

        if len(joint_feat) < seq_len:
            continue

        _rest = joint_feat.size(0) % seq_len
        if _rest > 0:
            _pruned_joint_feat = joint_feat[:-_rest, :]
        else:
            _pruned_joint_feat = joint_feat
        
        # chunking
        _chunk = _pruned_joint_feat.size(0) // seq_len
        
        _chunk_feat = torch.chunk(_pruned_joint_feat, _chunk, dim = 0)
        
        _chunk_feat = torch.stack(_chunk_feat)
        _chunk_feat = _chunk_feat
        z_list.append(_chunk_feat)
            
    return z_list


class SignProdDatasetPhoenix(Dataset):                                                                                                                                                                        
                                                                                                                                                                                                            
    def __init__(self, dataset_root):                                                   

        self.dataset_root = dataset_root
        self.samples = read_phoenix_data(self.dataset_root)
        self.samples = [torch.Tensor(sample) for sample in self.samples]
        self.samples = [torch.flatten(sample, start_dim = 1) for sample in self.samples]
        self.samples = chunkenize(self.samples, seq_len = 64)
        self.samples = [i for s in self.samples for i in s]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.Tensor(self.samples[idx])

def main():
    dataset_root = "/srv/storage/datasets/thiagocoutinho/datasets/phoenix/train"
    dataset = SignProdDatasetPhoenix(dataset_root, None, None)

if __name__ == "__main__":
    main()






    
