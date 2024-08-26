from cmath import isnan
import os
import sys
import json
import pickle
import math
import numpy as np
import pandas as pd

def read_pickle(file_fp):
    with open(file_fp, "rb") as handler:
        return pickle.load(handler)

def build_emotions_map():
    return dict(
        angry = ["anger"],
        disgust = ["disgust"],
        fear = ["fear"],
        happy = ["joy", "love", "optimism"],
        sad = ["sadness", "hopeless"],
        surprise = ["surprise"],
        neutral = ["trust", "anticipation"]
    )

def verify_type(instance):
    if isinstance(instance, str):
        print("IS LIST")
        return True

    if math.isnan(instance):
        print("IS NAN")
        return False
    

def main():

    df_path = "/srv/storage/datasets/rafaelvieira/code/SpanEmo/phrases/classified_phrases.csv"
    classified_videos_root = "/srv/storage/datasets/rafaelvieira/emotion_classification"
    spanemo_emotions = "anger,anticipation,disgust,fear,joy,love,optimism,hopeless,sadness,surprise,trust"
    df = pd.read_csv(df_path, sep = ";")
    emotions_map = build_emotions_map()
    matched_instances = list()
    matched_idxs = list()
    matched_emotions = list()

    for index, row in df.iterrows():

        print("Processing index {}".format(index))
        instance_name = row["instances_names"]
        classified_instance_fp = os.path.join(classified_videos_root, instance_name)
        if os.path.isfile(classified_instance_fp):
            instance_data = read_pickle(classified_instance_fp)
            text_emotions = row["predictions"]
            has_correct_type = verify_type(text_emotions)

            if has_correct_type:
                text_emotions = text_emotions.split(",")
                idxs = set()
                emotions_lst = list()
                found_match = False
                for idx, frame in enumerate(instance_data):
                    if frame:
                        detection = frame[0]
                        face_emotion = detection["emo_label"]
                        mapped_emotion = emotions_map[face_emotion]
                        for emotion in mapped_emotion:
                            if emotion in text_emotions:
                                found_match = True
                                #matched_instances.append(instance_name.replace(".pkl", ""))
                                idxs.add(idx)
                                emotions_lst.append((idx, text_emotions, face_emotion))
                if found_match:
                    matched_instances.append(instance_name.replace(".pkl", ""))
                    matched_idxs.append(idxs)
                    matched_emotions.append(emotions_lst)

    new_matches = list()
    for match in matched_idxs:
        if len(match) > 35:
            new_matches.append(match)

    import pdb
    pdb.set_trace()







if __name__ == "__main__":
    main()