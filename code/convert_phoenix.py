import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from natsort import natsorted
from deep_translator import GoogleTranslator


def translate_text(text):
    translated_text = GoogleTranslator(source="de", target="en").translate(text)
    return translated_text


def read_json(json_fp):
    with open(json_fp, "r") as handler:
        return json.load(handler)


def read_coords(points):
    coords = list()
    for idx in range(0, len(points), 3):
        x = points[idx]
        y = points[idx + 1]
        coords.append((x, y))
    return np.asarray(coords)


def main():

    annotations_root = "/home/rafael/masters/PHOENIX-2014-T-release-v3/dev"
    text_root = "/home/rafael/masters/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
    df = pd.read_csv(text_root, sep = "|")
    data = dict()
    
    for root, dirs, files in os.walk(annotations_root, topdown = False):
        try:
            instance_name = root.split(os.path.sep)[-1]
            if len(df.loc[df["name"] == instance_name]["translation"].tolist()) > 0:
                print("Running for instance name {}".format(instance_name))
                text = df.loc[df["name"] == instance_name]["translation"].tolist()[0]
                text = translate_text(text)
                keypoints = list()

                for file_ in natsorted(files):
                    annotation_fp = os.path.join(root, file_)
                    annotation = read_json(annotation_fp)
                    person = annotation["people"][0]
                    face_keypoints_2d = person["face_keypoints_2d"]
                    face_keypoints_2d = read_coords(face_keypoints_2d)
                    keypoints.append(face_keypoints_2d)
                
                data[instance_name] = dict()
                data[instance_name]["text"] = text
                data[instance_name]["kps"] = keypoints
        except:
            import traceback
            print(traceback.format_exc())
            continue
        

    with open("phoenix_data_dev.pkl", "wb") as handler:
        pickle.dump(data, handler)


if __name__ == "__main__":
    main()