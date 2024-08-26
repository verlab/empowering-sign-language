"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle

def _read_makeittalk_instance(instance_fp):
    instance = np.load(instance_fp)
    return instance

def _read_instance(instance_fp):
        with open(instance_fp, "rb") as handler:
                instance = pickle.load(handler)
        return instance["kps"]

def _read_pt_instance(instance_fp):
    instance = np.load(instance_fp)
    return instance

def add_wider_eye(fls):
        for fl in fls:
                fl[[37,38,43,44], 1] -=1
                fl[[40,41,46,47], 1] +=1
        return fls

def pull_mouth(fls):
        for fl in fls:
                fl[[48, 60],  0] -= 3
                fl[[54, 64],  0] += 3
                fl[[49, 50, 51, 52, 53],  0] -= 2
                fl[[55, 56, 57, 58, 59],  0] += 2
        return fls

def boost(fls, min_idx, max_idx, eyebrow_idx, increase = True):
        for idx in range(min_idx, max_idx):
                if idx > 63:
                        continue
                fls[idx][eyebrow_idx][1] -= 2
        return fls

def boost_eyebrows(fls):
        eyebrows_idx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        for eyebrow_idx in eyebrows_idx:
                for time_idx in range(1, fls.shape[0], 2):
                        previous_eyebrow = fls[time_idx - 1]
                        current_eyebrow = fls[time_idx]

                        movement_diff = current_eyebrow[eyebrow_idx, 1] - previous_eyebrow[eyebrow_idx, 1]
                        print(movement_diff)
                        if abs(movement_diff) > 0.3:
                                if movement_diff > 0:
                                        fls[time_idx][eyebrow_idx][1] += 0.4
                                else:
                                        fls[time_idx][eyebrow_idx][1] -= 0.4
                                                        
        return fls

def _read_instance_gt(instance_fp):
    with open(instance_fp, "rb") as handler:
        instance = pickle.load(handler)
    return instance

# Gabi ->> alterar parâmetros abaixo
default_head_name = 'template2' # essa é a imagem que será usada pra plotar, deve estar em /srv/storage/datasets/gabrielaneme/code/codeMakeItTalk/MakeItTalk/examples, formato 256x256
input_npz_folder = "outputs_reprotest" # diretório do caminho npz (entrada)
output_mp4_folder = "landmarks_videos" # nome e onde salvar vídeo de saída
keypoint_type = "ours"
os.makedirs(output_mp4_folder, exist_ok=True)
#fzDHRCKr7wU_8-8-rgb_front
#fzDHRCKr7wU_8-8-rgb_front_rendered

ADD_NAIVE_EYE = True
CLOSE_INPUT_FACE_MOUTH = False


parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name))
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=2.)
parser.add_argument('--amp_lip_y', type=float, default=2.)
parser.add_argument('--amp_pos', type=float, default=.5)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')

parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')

opt_parser = parser.parse_args()

img =cv2.imread("face_image.jpg")

files_failed = []
#allowlist = ["02yXvi1VmPI_12-8-rgb_front", "exoR8fu9q4M_18-8-rgb_front", "aomlhMR9PLQ_8-8-rgb_front"]
#allowlist = ["aomlhMR9PLQ_8-8-rgb_front"]
#allowlist = ["fzDHRCKr7wU_8-8-rgb_front.npz"]
allowlist = ["fZgWKh3ENoE_4-8-rgb_front.npz", "fzncPNr2Sc0_3-8-rgb_front.npz", "fzDHRCKr7wU_10-8-rgb_front.npz"]
model = Image_translation_block(opt_parser, single_test=True)

for root, dirs, files in os.walk(input_npz_folder, topdown = False):
        #files_filtered = [file_ for file_ in files if file_.endswith(".pkl")]
        #files = [file_ for file_ in files if file_ in allowlist]
        for file_ in files:
                print("Rendering video for {}".format(file_))
                complete_in_dir = os.path.join(root, file_)
                out_name = file_.replace(".npz", "_rendered.mp4")
                complete_out_dir = os.path.join(output_mp4_folder, out_name)
                if os.path.isfile(complete_in_dir):

                        if keypoint_type == "mt":
                                fl = _read_makeittalk_instance(complete_in_dir)
                                fl = np.append(fl, np.zeros((fl.shape[0], 68, 1)), axis = 2)
                                fl = fl * 256
                        elif keypoint_type == "gt":
                                fl = np.asarray(_read_instance(complete_in_dir))
                                fl = fl[:,:-1,:] 
                                fl = np.append(fl, np.zeros((fl.shape[0], 68, 1)), axis = 2)
                                fl = fl * 256
                        elif keypoint_type == "pt":
                                fl = _read_pt_instance(complete_in_dir)[:, :-1].reshape(-1, 68, 2)
                                fl = np.append(fl, np.zeros((fl.shape[0], 68, 1)), axis = 2)
                                fl = fl * 256
                        else:
                                npz_file = np.load(complete_in_dir)
                                kps = npz_file["arr_0"].squeeze(0)
                                #kps = _read_instance(complete_in_dir)

                                #fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
                                fl = np.asarray(kps)
                                fl = fl[:,:-1,:]
                                fl = fl * 256
                                fl = add_wider_eye(fl)
                                fl = pull_mouth(fl)
                                #fl = boost_eyebrows(fl)
                                #fl = util.add_naive_eye(fl)
                                fl = np.append(fl, np.zeros((fl.shape[0], 68, 1)), axis = 2)
                                #fl = util.add_naive_eye(fl)
                        try:
                                # ''' STEP 6: Imag2image translation '''
                                with torch.no_grad():
                                        model.single_test(jpg=img, fls=fl, filename="out.npz", prefix=opt_parser.jpg.split('.')[0], instance_name = file_)
                                #shutil.move('/home/rafael/data/datasets/out.mp4', complete_out_dir)
                        except:
                                import traceback
                                print(traceback.format_exc())
                                files_failed.append(file_)

representative_figure = np.vstack((model.row1, model.row2, model.row3))
cv2.imwrite("/app/frames_root/representative_figure.png", representative_figure)
