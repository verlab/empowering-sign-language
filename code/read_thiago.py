import os
import torch
import pdb
import numpy as np

def read_data(data_path):

    ## Load model data
    data = torch.load(data_path)
    generated = data['outputs']
    reference = data['reference']

    ## Cut references and generated data to both have same length
    generated = [gen[:len(ref)] if len(gen) > len(ref) else gen for gen, ref in zip(generated, reference)]
    reference = [ref[:len(gen)] if len(ref) > len(gen) else ref for gen, ref in zip(generated, reference)]

    return generated, reference

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

def format_data(data):
    generated = [torch.Tensor(gen) for gen in data]
    generated_flatten = [torch.flatten(sample, start_dim = 1) for sample in generated]
    generated_chunks = chunkenize(generated_flatten, seq_len = 64)
    generated_chunks_flatten = [i for s in generated_chunks for i in s]
    generated_chunks_flatten = [np.array(g) for g in generated_chunks_flatten]
    return generated_chunks_flatten


def save_predictions(output_fp, faces_fake):
    np.savez(output_fp, faces_fake)

def write_data(data, output_root):
    chunk_pattern = "chunk_{}.npz"
    os.makedirs(output_root, exist_ok=True)
    for idx, d in enumerate(data):
        print("Writing instance: {}".format(idx))
        output_fp = os.path.join(output_root, chunk_pattern.format(idx))
        save_predictions(output_fp, d)


def main():
    output_path = "/srv/storage/datasets/thiagocoutinho/experiments/phoenix/NSLPG/narslp/1x5fty2l/test_outputs/outputs.pt"
    formatted_instances_ours = "/srv/storage/datasets/rafaelvieira/thiago_ours"
    formatted_instances_gt = "/srv/storage/datasets/rafaelvieira/thiago_gt"

    generated, reference = read_data(output_path)

    generated_formatted = format_data(generated)
    reference_formatted = format_data(reference)

    write_data(generated_formatted, formatted_instances_ours)
    write_data(reference_formatted, formatted_instances_gt)


if __name__ == "__main__":
    main()