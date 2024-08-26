import cv2
import os
import uuid
import numpy as np
import torch
import torch.nn as nn

class Utils(object):
    X_MAX_DIM = 256
    Y_MAX_DIM = 256

    @staticmethod
    def weights_init(m):
        if hasattr(m, 'weight') and m.weight.requires_grad:
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        else:
            print("Layer does not need grad, thus not initializing")
    
    @staticmethod
    def draw_keypoints(kps):
        img = np.ones((256, 256, 3))*255
        height = Utils.Y_MAX_DIM
        width = Utils.X_MAX_DIM
        for kp in kps:
            x = kp[0]
            y = kp[1]
            img = cv2.circle(img, (int(x*width), int(y*height)), 2, (0, 255 ,0), 2)

        return img

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    @staticmethod
    def plot_video(predicted, gts, video_file):

        FPS = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (512, 256), True)
        
        num_frames = 0

        for idx, (pred, gt) in enumerate(zip(predicted, gts)):
            frame = np.concatenate((pred, gt), axis=1)
            video.write(frame.astype(np.uint8))
            num_frames += 1

        video.release()

    @staticmethod
    def plot_video_single(predicted, video_file):
        FPS = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (256, 256), True)
        num_frames = 0
        for idx, pred in enumerate(predicted):
            
            video.write(pred.astype(np.uint8))
            num_frames += 1

        video.release()
    
    @staticmethod
    def visualize_data_single(predicted, video_file):
        for pred in predicted:
            predicted_imgs = [Utils.draw_keypoints(kp) for kp in pred]
            Utils.plot_video_single(predicted_imgs, video_file)


    @staticmethod
    def visualize_data(fake_faces, kps, out_root, step, file_names):
        print("Plotting data...")

        fake_faces = fake_faces.permute(0, 2, 3, 1)
        kps = kps.permute(0, 2, 3, 1)

        for idx, (predicted, gt, file_name) in enumerate(zip(fake_faces.cpu().numpy(), kps.cpu().numpy(), file_names)):            
            out_folder = os.path.join(out_root, "step_{}".format(step))
            os.makedirs(out_folder, exist_ok = True)
            out_fp = os.path.join(out_folder, "{}.mp4".format(file_name))
            predicted_imgs = [Utils.draw_keypoints(kp) for kp in predicted]
            gts_imgs = [Utils.draw_keypoints(kp) for kp in gt]
            Utils.plot_video(predicted_imgs, gts_imgs, out_fp)

    @staticmethod
    def save_landmarks(output_root, file_names, fake_faces):

        fake_faces = fake_faces.permute(0, 2, 3, 1)
        os.makedirs(output_root, exist_ok = True)
        for idx, (predicted, file_name) in enumerate(zip(fake_faces.cpu(), file_names)):
            out_folder = os.path.join(output_root, "{}.npz".format(file_name))
            np.savez(out_folder, predicted)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor