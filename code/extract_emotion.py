import os
import cv2
import sys
import re
import pickle
from rmn import RMN

def get_speaker(video_name):
    try:
        speaker_id = video_name[14:]
        speaker_id = re.search(r'\d+', speaker_id).group()
    except:
        import traceback
        print(traceback.format_exc())

    return speaker_id

def get_frames(video_name):      
    vid = cv2.VideoCapture(video_name)
    success = True
    frames = list()
    while success:
        success, image = vid.read()
        frames.append(image)
    return frames

def write_pickle(data, output_fp):
    with open(output_fp, 'wb') as handler:
        pickle.dump(data, handler, protocol=pickle.HIGHEST_PROTOCOL)
  

def main():

    videos_root = sys.argv[1]
    
    videos = os.listdir(videos_root)
    output_root = "/srv/storage/datasets/rafaelvieira/emotion_classification"
    os.makedirs(output_root, exist_ok = True)
    m = RMN()

    for idx, video in enumerate(videos):
        video_fp = os.path.join(videos_root, video)
        output_fp = os.path.join(output_root, video.replace("mp4", "pkl"))
        speaker_id = get_speaker(video_fp.split(os.path.sep)[-1])
        if speaker_id == "8":
            print("PROCESSING VIDEO IDX: {}".format(idx))
            video_emotions = list()
            frames = get_frames(video_fp)
            for frame in frames:
                try:
                    results = m.detect_emotion_for_single_frame(frame)
                    video_emotions.append(results)
                except:
                    pass
            write_pickle(video_emotions, output_fp)


if __name__ == "__main__":
    main()