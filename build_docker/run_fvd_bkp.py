# Run FVD for a series of videos


import os
import skvideo.io 
import numpy as np
from glob import glob
import tensorflow.compat.v1 as tf
import frechet_video_distance as fvd

def get_paths(path):
    video_paths = glob(path + "/*")
    if len(video_paths) < 16:
        if len(video_paths) % 8 == 0: video_paths = video_paths * 2
        elif len(video_paths) % 4 == 0: video_paths = video_paths * 4
        elif len(video_paths) % 2 == 0: video_paths = video_paths * 8
        else: video_paths = video_paths * 16
    
    video_paths = video_paths[:len(video_paths)//16 * 16]
    return video_paths


def getVideos(video_paths):
    """Returns a compact numpy array containing all the videos frames in the path given.
    :param
    path (string): full path the video dataset  in the form of:
                            path_gen/folder_name_1/frame1.ext  path_gen/folder_name_1/frame2.ext ...
                            path_gen/folder_name_2/frame1.ext  path_gen/folder_name_2/frame2.ext ...
                            path_gen/folder_name_3/frame1.ext  path_gen/folder_name_3/frame2.ext ...
    :returns
    videos : a numpy array containing all videos. The shape is [number_of_videos, frames, frame_height, frame_width, 3]
    the number_of_videos has to be multiple of 16 as indicated by the authors of the original repo
    """
    videos = []

    for vp in video_paths:
        print(vp)
        videodata = skvideo.io.vread(vp)
        videos.append(videodata)
    videos = np.asarray(videos)
    return videos


def get_FVD(path_real, path_gt):

    paths_gen = get_paths(path_real)
    paths_og = [i.replace(path_real, path_gt) for i in paths_gen]


    real_videos = getVideos(paths_gen)
    gt_videos = getVideos(paths_og)

    with tf.Graph().as_default():
        
        fvd_res = []

        for i in range(0, real_videos.shape[0], 16):
            print("Forwarding idx {}".format(i))
            real_videos_b = real_videos[i:i+16]
            gt_videos_b = gt_videos[i:i+16]

            _real_videos = tf.convert_to_tensor(real_videos_b)
            _gt_videos = tf.convert_to_tensor(gt_videos_b)

            _real_videos = fvd.preprocess(_real_videos,(224, 224))
            _gt_videos = fvd.preprocess(_gt_videos,(224, 224))
            
            result = fvd.calculate_fvd(
                fvd.create_id3_embedding(_real_videos),
                fvd.create_id3_embedding(_gt_videos)
                )

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                ress = sess.run(result)
                print("FVD is: %.2f." % ress)

            fvd_res.append(ress)

    print("FVD mean = ", sum(fvd_res)/len(fvd_res))


if __name__ == '__main__':

    path_real = "/app/videos_volume"
    path_gt =  "/app/test_rendered"
    
    get_FVD(path_real, path_gt)
