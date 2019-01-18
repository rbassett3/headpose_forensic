import os
from skimage import io
from face_proc_v2 import FaceProc
import cv2
import argparse
import yaml

from utils import pose_utils as pu
import pickle
import numpy as np
from test_v2 import exam_img, exam_video
import pandas as pd

def main(args):
    all_paths = os.listdir(args.input_dir)
    proba_list = []

    # initiate face process class, used to detect face and extract landmarks
    face_inst = FaceProc()

    # initialize SVM classifier for face forensics
    with open(args.classifier_path, 'rb') as f:
        model = pickle.load(f)
    classifier = model[0]
    scaler = model[1]

    for f_name in all_paths:
        f_path = os.path.join(args.input_dir, f_name)
        print('_'*20)
        print('Testing: ' + f_name)
        suffix = f_path.split('.')[-1]
        if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp']:
            proba, optout = exam_img(args, f_path, face_inst, classifier, scaler)
        elif suffix.lower() in ['mp4', 'avi', 'mov', 'mts']:
            proba, optout = exam_video(args, f_path, face_inst, classifier, scaler)
        print('fake_proba: {},   optout: {}'.format(str(proba), optout))
        tmp_dict = dict()
        tmp_dict['file_name'] = f_name
        tmp_dict['probability'] = proba
        tmp_dict['optout'] = optout
        proba_list.append(tmp_dict)
    pickle.dump(proba_list, open('proba_list.p', 'wb'))


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="say something you like")
   parser.add_argument('--cfg', type=str, default='cfgs/head_pose_cfg/head_pose_B.yaml')
   parser.add_argument('--input_dir', type=str, default='debug_data')
   parser.add_argument('--markID_c', type=str, default='1-36,49,55', help='landmark ids to estimate CENTRAL face region')
   parser.add_argument('--markID_a', type=str, default='18-36,49,55', help='landmark ids to estimate WHOLE face region')
   # parser.add_argument('--classifier_path', type=str, default='models/trained_models/40SVM_rbf_default_r&t_converted.p')
   parser.add_argument('--classifier_path', type=str, default='models/trained_models/R_full_mat_t_vec_model.p')
   parser.add_argument('--save_file', type=str, default='result/final_results.p')
   args = parser.parse_args()
   main(args)