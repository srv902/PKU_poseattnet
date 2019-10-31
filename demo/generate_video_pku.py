##################################
#
# PKU demo generator
# Written by: Saurav Sharma
#
# ffmpeg used to convert frames to video || -->  ffmpeg -r 30 -i modified_frames/0319-M/%5d.jpg -vb 20M myvideo.mpg
#
#
##################################

import pandas as pd
import numpy as np
import pickle
import cv2
from collections import OrderedDict
import os
import argparse
from shutil import copyfile

IMAGE_PATH = '/data/stars/user/sasharma/PKU_poseattnet/demo/original_frames'
SAVE_PATH = '/data/stars/user/sasharma/PKU_poseattnet/demo/modified_frames'
ACTION_LABEL_FILE = '../data/pku_mmd_label_map.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='Parser for generating demo')
    parser.add_argument('--videoname', type=str, help='videoname for generating demo')
    args = parser.parse_args()
    return args

def get_class_name():
    actions = pd.read_csv(ACTION_LABEL_FILE, header=None)
    classname = {}
    for i in range(actions.shape[0]):
        classname[i+1] = actions.iloc[i,0]
    return classname

def prepare_labels(gtfile, predfile, classnames):
    gtfile['type'] = 'gt'
    predfile['type'] = 'pred'
    newdf = pd.concat([gtfile, predfile], ignore_index=True)
    newdf = newdf.sort_values('t-start')
    min1 = newdf['t-start'].values.min()
    max1 = newdf['t-end'].values.max()
    label_dict = OrderedDict()
    for i in range(newdf.shape[0]):
        start, end, label, conf, type = newdf.iloc[i, 1], newdf.iloc[i, 2], classnames[newdf.iloc[i, 3]], round(newdf.iloc[i, 4], 3), newdf.iloc[i, 5]

        for j in range(start, end+1):
            if j not in label_dict:
                label_dict[j] = []
                label_dict[j].append([label, conf, type])
            else:
                label_dict[j].append([label, conf, type])

    return label_dict

def create_demo(label, vidname):
    print(">>>>>>Overlay predictions and ground truth on frames<<<<<")
    print(vidname)
    if not os.path.isdir(os.path.join(SAVE_PATH, vidname)):
        os.mkdir(os.path.join(SAVE_PATH, vidname))

    # assuming starting frame number for all videos is 1
    startidx, numframes = 1, len(os.listdir(os.path.join(IMAGE_PATH, vidname)))
    for i in range(startidx, numframes):
        frameid = "%05d" % i
        srcfileid = os.path.join(IMAGE_PATH, vidname, str(frameid)+'.jpg')
        destfileid = os.path.join(SAVE_PATH, vidname, str(frameid)+'.jpg')

        img = cv2.imread(srcfileid)
        win_H, win_W = 540, 640
        window = np.full((win_H, win_W, 3), 210, np.uint8)
        new_window = window.copy()
        start = win_H - img.shape[0]
        end = win_W - img.shape[1]
        new_window[0:img.shape[0], end:win_W, :] = img
        labelpos = 500
        predpos = 130
        gtpos = 112

        if i not in label.keys():
            cv2.putText(new_window, "Prediction: Background", (predpos, labelpos + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
            cv2.putText(new_window, "Groundtruth: Background", (gtpos, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
            # copyfile(srcfileid, destfileid)
        else:
            # print("Updating the frame with gt and pred")

            tolabel = label[i]

            pred1 = [i for i in tolabel if i[2] == 'pred']
            gt1 = [i for i in tolabel if i[2] == 'gt']

            if len(gt1) == 1 and len(pred1) == 1:
                if len(pred1[0][0]) > 30:
                    # print("big text")
                    predpos = 70
                    gtpos = 52

                if pred1[0][0] == gt1[0][0]:
                    cv2.putText(new_window, "Prediction: " + pred1[0][0] + '(' + str(pred1[0][1]) + ')', (predpos, labelpos+25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
                    cv2.putText(new_window, "Groundtruth: " + gt1[0][0], (gtpos, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
                else:
                    cv2.putText(new_window, "Prediction: " + pred1[0][0] + '(' + str(pred1[0][1]) + ')', (predpos, labelpos + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 153), 1)
                    cv2.putText(new_window, "Groundtruth: " + gt1[0][0], (gtpos, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
            elif len(gt1) == 0 and len(pred1) == 1:
                if len(pred1[0][0]) > 30:
                    # print("big text")
                    predpos = 70
                    gtpos = 52
                cv2.putText(new_window, "Prediction: " + pred1[0][0] + '(' + str(pred1[0][1]) + ')', (predpos, labelpos+25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 153), 1)
                cv2.putText(new_window, "Groundtruth: Background", (gtpos, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
            elif len(gt1) == 1 and len(pred1) == 0:
                if len(gt1[0][0]) > 30:
                    # print("big text")
                    predpos = 70
                    gtpos = 52
                cv2.putText(new_window, "Prediction: Background", (predpos, labelpos + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 153), 1)
                cv2.putText(new_window, "Groundtruth: " + tolabel[0][0], (gtpos, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)

        cv2.imwrite(destfileid, new_window)


# prediction demo generator below
if __name__ == "__main__":
    args = parse_args()
    # pickle filenames for gt and pred
    gtfilename = "ground_truth_dfnew.pkl"
    predfilename = "merged_predictions.pkl"

    # load gt and pred pickle files
    gtfile = pickle.load(open(gtfilename, 'rb'))
    predfile = pickle.load(open(predfilename, 'rb'))

    # select the video whose demo is desired
    videotofilter = args.videoname
    gtselect = gtfile[gtfile['video-id'] == videotofilter]
    predselect = predfile[predfile['video-id'] == videotofilter]

    # filtering out background..
    predselect = predselect[predselect['label'] != 0]

    print("Selected groud truth")
    print(gtselect)

    print("Selected prediction")
    print(predselect)

    # get class name for the action label values
    classnames = get_class_name()

    label_annot = prepare_labels(gtselect, predselect, classnames)
    # print("unified table of pred and gt")
    # print(label_annot)

    create_demo(label_annot, videotofilter)
