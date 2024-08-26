import os
import sys
import json
import io
import numpy as np

from einops import rearrange

def read_phoenix_data(data_path, skip_frames=1):

    #print(f"Loading phoenix {data_path.split('/')[-1]} set")

    trg_size = 361
    samples = []    
    trg_path = f"{data_path}.skels"
    line_count = 0
    with io.open(trg_path, mode='r', encoding='utf-8') as trg_file:     

        i = 0

        for trg_line in trg_file:
            import pdb
            pdb.set_trace()
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


def main():

    signer_n = 1
    signers_root = "/srv/storage/datasets/rafaelvieira/signers_phoenix/signer_{}_train".format(signer_n)
    samples = read_phoenix_data(signers_root)
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()