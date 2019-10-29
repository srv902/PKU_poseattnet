##################################
#
# PKU demo generator
# Written by: Saurav Sharma
#
##################################

import pandas as pd
import numpy as np
import pickle
import cv2
from collections import OrderedDict
import os
from shutil import copyfile

IMAGE_PATH = '/data/stars/user/sasharma/PKU_poseattnet/demo/original_frames'
SAVE_PATH = '/data/stars/user/sasharma/PKU_poseattnet/demo/modified_frames'
ACTION_LABEL_FILE = '../data/pku_mmd_label_map.txt'


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
        if i not in label.keys():
            copyfile(srcfileid, destfileid)
        else:
            # print("Updating the frame with gt and pred")
            img = cv2.imread(srcfileid)

            #"""
            win_H, win_W = 540, 640
            window = np.full((win_H, win_W, 3), 210, np.uint8)
            # print("window shape ",window.shape)
            new_window = window.copy()
            start = win_H - img.shape[0]
            end = win_W - img.shape[1]
            new_window[0:img.shape[0], end:win_W, :] = img
            # """
            # new_window = img
            labelpos = 500
            tolabel = label[i]

            pred1 = [i for i in tolabel if i[2] == 'pred']
            gt1 = [i for i in tolabel if i[2] == 'gt']

            if len(gt1) == 1 and len(pred1) == 1:
                cv2.putText(new_window, "Prediction: " + pred1[0][0] + '(' + str(pred1[0][1]) + ')', (100, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 102, 0), 1)
                cv2.putText(new_window, "Actual: " + gt1[0][0], (130, labelpos + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            elif len(gt1) == 0 and len(pred1) == 1:
                cv2.putText(new_window, "Prediction: " + pred1[0][0] + '(' + str(pred1[0][1]) + ')', (100, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 153), 1)
                cv2.putText(new_window, "Actual: Background", (130, labelpos + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            elif len(gt1) == 1 and len(pred1) == 0:
                cv2.putText(new_window, "Actual: " + tolabel[0][0], (130, labelpos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            cv2.imwrite(destfileid, new_window)


# prediction demo generator below
if __name__ == "__main__":
    # pickle filenames for gt and pred
    gtfilename = "ground_truth_dfnew.pkl"
    predfilename = "merged_predictions.pkl"

    # load gt and pred pickle files
    gtfile = pickle.load(open(gtfilename, 'rb'))
    predfile = pickle.load(open(predfilename, 'rb'))

    # select the video whose demo is desired
    videotofilter = '0319-M'
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