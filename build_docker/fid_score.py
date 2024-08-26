import sys
sys.path.append("..")

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from scipy import linalg
from models.pose_autoencoder import EmbeddingNet

import pickle
import random

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch_size', '-bs', type = int, default = 24)

parser.add_argument('--device', '-dev', type = int, default = 0)

parser.add_argument('--path_gt', '-pgt', type = str, default = "test_repro/test")

parser.add_argument('--path_generated', '-pg', type = str, default = "outputs_reprotest")
parser.add_argument('--keypoint_type', '-kt', type = str, default = "ours")

parser.add_argument('--path_checkpoints', '-pck', type = str, default = "checkpoints_ae2/step_70300/autoencoder.pth")


def _read_instance_gt(root, instance_name):
    instance_fp = os.path.join(root, instance_name)
    with open(instance_fp, "rb") as handler:
        instance = pickle.load(handler)
    return instance

def _read_makeittalk_instance(root, instance_name):
    instance_fp = os.path.join(root, instance_name)
    instance = np.load(instance_fp)[:, :, :-1]
    return instance

def _read_ours_instance(root, instance_name):
    instance_fp = os.path.join(root, instance_name)
    instance = np.load(instance_fp)["arr_0"]
    return instance

def _read_pt_instance(baseline_root, instance_name):
    instance_fp = os.path.join(baseline_root, instance_name)
    instance = np.load(instance_fp)
    return instance

def get_activations(my_instances, model, batch_size=50, device='cpu'):
    """Calculates the activations of the latente layer for keypoint faces.

    Params:
    -- my_instances: List of keypoints instances [64,136]
    -- model       : Instance of autoencoder
    -- batch_size  : Batch size of instances for the model to process at once.
                     
    -- device      : Device to run calculations

    Returns:
    -- A numpy array of dimension (num instances, dims) that contains the
       latente layer activations of the given tensor when feeding autoencoder with the
       query tensor.
    """

    model.eval()

    pred_arr = np.empty((len(my_instances), 32))

    start_idx = 0
    
    for batch_start in range(0, len(my_instances), batch_size):  

        batch_final = (batch_start + batch_size) if (batch_start + batch_size) < len(my_instances) else len(my_instances)
        batch = np.asarray(my_instances[batch_start:batch_final])    
        batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float)

        with torch.no_grad():
            feat, a, b, recon_poses = model(batch_tensor)


        pred = feat.cpu().detach().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        # Numerical error might give slight imaginary component
        #if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #    m = np.max(np.abs(covmean.imag))
        #    raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(my_instances, model, batch_size=50, device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(my_instances, model, batch_size, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, keypoint_type, device):

    my_instances = list()

    # todo: format mt and pt keypoints
    for instance_name in sorted(os.listdir(path)):
        try:
            if keypoint_type == "gt": 
                my_instances.append(np.asarray(_read_instance_gt(path, instance_name)["kps"])[:, :-1, :].reshape(-1, 136))
            elif keypoint_type == "mt":
                my_instances.append(_read_makeittalk_instance(path, instance_name).reshape(-1, 136))
            elif keypoint_type == "pt":
                my_instances.append(_read_pt_instance(path, instance_name)[:,:-1].reshape(-1, 136))
            elif keypoint_type == "ours":
                my_instances.append(_read_ours_instance(path, instance_name).squeeze(0)[:, :-1, :].reshape(-1, 136))
            elif keypoint_type == "nslp":
                test = _read_instance_gt(path, instance_name)/256
                test = test.reshape(-1, 136).numpy()
                my_instances.append(test)
        except:
            print("found no existing instance, skipping")
            import traceback
            print(traceback.format_exc())

    m, s = calculate_activation_statistics(my_instances, model, batch_size, device)

    return m, s


def calculate_fid_given_paths(paths_gt, paths_g, batch_size, device, keypoint_type,path_checkpoints):

    """Calculates the FID of two paths"""
    if not os.path.exists(paths_gt):
        raise RuntimeError('Invalid path: %s' % paths_gt)

    if not os.path.exists(paths_g):
        raise RuntimeError('Invalid path: %s' % paths_g)

    

    # pose_dim = 68*2,n_frames

    model = EmbeddingNet(136, 64).to(device)

    try:
        model.load_state_dict(torch.load(path_checkpoints, map_location=torch.device("cpu")))
    except:
        import traceback
        print(traceback.format_exc())
        import pdb
        pdb.set_trace()


    model.eval()

    m1, s1 = compute_statistics_of_path(paths_gt, model, batch_size,
                                        "gt", device)

    m2, s2 = compute_statistics_of_path(paths_g, model, batch_size,
                                        keypoint_type, device)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():

    args = parser.parse_args()
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    #device = torch.device('cuda:{}'.format(args.device)) if args.device != -1 else torch.device('cuda')
    device = torch.device("cpu")

    
    fid_value = calculate_fid_given_paths(args.path_gt,
                                          args.path_generated,
                                          args.batch_size,
                                          device,
                                          args.keypoint_type,
                                          args.path_checkpoints)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
