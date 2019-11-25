# ***************************************************
# PKU evaluation code customized for action detection
# ***************************************************

import os
import pickle
import numpy as np
import pandas as pd
import collections
number_label = 52
import matplotlib.pyplot as plt

def prepare_prediction(data, name=''):
    # predictions = pd.DataFrame(columns=['video-id', 't-start', 't-end', 'label'])

    # for i in test_split:
    #     tmp = {'video-id': i[0], 't-start': i[2], 't-end': i[3], 'label': None}
    #     predictions = predictions.append(tmp, ignore_index=True)

    videoid = [i[0] for i in test_split]
    tstart = [i[2] for i in test_split]
    tend = [i[3] for i in test_split]
    label = [None for i in test_split]

    data = {'video-id': videoid, 't-start': tstart, 't-end': tend, 'label': label}
    predictions = pd.DataFrame.from_dict(data)
    # predictions = predictions[['video-id', 't-start', 't-end', 'label']]

    pickle.dump(predictions, open('predictions_'+name+'df.pkl', 'wb'))

def get_distinct_segments(cls_probs, predictions):
    new_prediction = pd.DataFrame(columns=['video-id', 't-start', 't-end', 'label', 'conf'])
    first_entry = True
    print("gamma ", gamma)
    for i in range(predictions.shape[0]):
        tstartcurr, tendcurr = predictions.iloc[i, 1], predictions.iloc[i, 2]
        if i < predictions.shape[0] - 1:
            tstartnext, tendnext = predictions.iloc[i + 1, 1], predictions.iloc[i + 1, 2]
            vidname, vidnamenext = predictions.iloc[i, 0], predictions.iloc[i + 1, 0]

            if vidname == vidnamenext and tendcurr - tstartnext == overlap:
                # print("prob shape ")
                # print(cls_probs[i, 1:].shape)
                avg_probs = np.average(np.vstack((cls_probs[i, :], cls_probs[i + 1, :])), axis=0)

                if avg_probs[0] > sum(avg_probs[1:]) or sum(avg_probs[1:]) < gamma:
                    label = 0
                    labelconf = avg_probs[0]
                else:
                    label = np.argmax(avg_probs[1:]) + 1
                    labelconf = avg_probs[label]

                if cls_probs[i, 0] > sum(cls_probs[i, 1:]) or sum(cls_probs[i, 1:]) < gamma:
                    seglabel = 0
                    segconf = cls_probs[i, 0]
                else:
                    seglabel = np.argmax(cls_probs[i, 1:]) + 1
                    segconf = cls_probs[i, seglabel]

                # break
                # add segment of 20 frames ..

                if first_entry:
                    tmp = {'video-id': vidname, 't-start': tstartcurr, 't-end': tstartnext,
                           'label': seglabel, 'conf': segconf}
                    new_prediction = new_prediction.append(tmp, ignore_index=True)
                    # below adds the overlap part
                    tmp = {'video-id': vidname, 't-start': tstartnext, 't-end': tendcurr, 'label': label,
                           'conf': labelconf}
                    new_prediction = new_prediction.append(tmp, ignore_index=True)
                    first_entry = False
                else:
                    tmp = {'video-id': vidname, 't-start': tstartcurr + overlap, 't-end': tstartnext,
                           'label': seglabel, 'conf': segconf}
                    new_prediction = new_prediction.append(tmp, ignore_index=True)
                    # below adds the overlap part
                    tmp = {'video-id': vidname, 't-start': tstartnext, 't-end': tendcurr, 'label': label,
                           'conf': labelconf}
                    new_prediction = new_prediction.append(tmp, ignore_index=True)
                    # tmp = {'video-id': vidname, 't-start': tendnext, 't-end': tendnext - overlap, 'label': predictions.iloc[i+1, 3], 'conf': predictions.iloc[i+1, 4]}
                    # new_prediction = new_prediction.append(tmp, ignore_index=True)
            elif vidname != vidnamenext:
                tmp = {'video-id': vidname, 't-start': tstartcurr + overlap, 't-end': tendcurr,
                       'label': seglabel, 'conf': segconf}
                new_prediction = new_prediction.append(tmp, ignore_index=True)
                print("End of the video " + vidname + " has been reached!! Moving to next video")
                first_entry = True
            elif tendcurr - tstartnext != overlap:
                print("Error in overlap calculation!!")
        else:
            if cls_probs[i, 0] > sum(cls_probs[i, 1:]) or sum(cls_probs[i, 1:]) < gamma:
                seglabel = 0
                segconf = cls_probs[i, 0]
            else:
                seglabel = np.argmax(cls_probs[i, 1:]) + 1
                segconf = cls_probs[i, seglabel]

            tmp = {'video-id': vidname, 't-start': tstartcurr + overlap, 't-end': tendcurr,
                   'label': seglabel, 'conf': segconf}
            new_prediction = new_prediction.append(tmp, ignore_index=True)

    return new_prediction

def get_distinct_segments_no_overlap(cls_probs, predictions):
    new_prediction = pd.DataFrame(columns=['video-id', 't-start', 't-end', 'label', 'conf'])
    print("gamma ", gamma)
    videoid, tstart, tend, label, conf = [], [], [], [], []
    for i in range(predictions.shape[0]):
        tstartcurr, tendcurr, vidname = predictions.iloc[i, 1], predictions.iloc[i, 2], predictions.iloc[i, 0]
        if cls_probs[i, 0] > sum(cls_probs[i, 1:]) or sum(cls_probs[i, 1:]) < gamma:
            seglabel = 0
            segconf = cls_probs[i, 0]
        else:
            seglabel = np.argmax(cls_probs[i, 1:]) + 1
            segconf = cls_probs[i, seglabel]

        # tmp = {'video-id': vidname, 't-start': tstartcurr, 't-end': tendcurr, 'label': seglabel, 'conf': segconf}
        # new_prediction = new_prediction.append(tmp, ignore_index=True)
        videoid.extend([vidname])
        tstart.extend([tstartcurr])
        tend.extend([tendcurr])
        label.extend([seglabel])
        conf.extend([segconf])

    data = {'video-id': videoid, 't-start': tstart, 't-end': tend, 'label': label, 'conf': conf}
    new_prediction = pd.DataFrame.from_dict(data)
    # new_prediction = new_prediction[['video-id', 't-start', 't-end', 'label', 'conf']]
    # print(new_prediction[new_prediction['video-id'] == '0291-L'])
    return new_prediction

def apply_multiscale_processing():
    # this is before merge prediction step where we take the best model for 32 frame split and do max of the scores for each of 16 frame segments
    scores32 = pickle.load(open('../output/PKU_GRU_clsprobs_test_by_video_epoch11_1024_LSTM_32.pkl', 'rb'))
    split32 = pickle.load(open('../output/validation_split_32frames.pkl','rb'))

    print("scores32 shape ", len(scores32))
    print("split32 len ", len(split32))



def get_merge_predictions(validation_videos, split_prediction):
    merged_predictions = pd.DataFrame(columns=['video-id', 't-start', 't-end', 'label', 'conf'])

    pred_groups = split_prediction.groupby('video-id')
    # print("pred groups ")
    # print(pred_groups.groups)

    idx = 0
    videoid, tstart, tend, label, conf = [], [], [], [], []
    for vid in validation_videos:
        f = pred_groups.get_group(vid)
        start = True
        # idx = 0
        for i in range(f.shape[0]):
            if start == True:
                tmp = {'video-id': f.iloc[i, 0], 't-start': f.iloc[i, 1], 't-end': f.iloc[i, 2], 'label': f.iloc[i, 3],
                       'conf': f.iloc[i, 4]}
                # merged_predictions = merged_predictions.append(tmp, ignore_index=True)
                videoid.extend([f.iloc[i,0]])
                tstart.extend([f.iloc[i,1]])
                tend.extend([f.iloc[i,2]])
                label.extend([f.iloc[i,3]])
                conf.extend([f.iloc[i,4]])
                start = False
            else:
                if f.iloc[i, 3] == label[idx]:  # merged_predictions.iloc[idx, 3]:
                    # merged_predictions.iloc[idx, 2] = f.iloc[i, 2]
                    # merged_predictions.iloc[idx, 4] = (f.iloc[i, 4] + merged_predictions.iloc[idx, 4]) * 0.5
                    tend[idx] = f.iloc[i, 2]
                    conf[idx] = (f.iloc[i, 4] + conf[idx]) * 0.5
                else:
                    tmp = {'video-id': f.iloc[i, 0], 't-start': f.iloc[i, 1], 't-end': f.iloc[i, 2],
                           'label': f.iloc[i, 3],
                           'conf': f.iloc[i, 4]}
                    # merged_predictions = merged_predictions.append(tmp, ignore_index=True)
                    videoid.extend([f.iloc[i, 0]])
                    tstart.extend([f.iloc[i, 1]])
                    tend.extend([f.iloc[i, 2]])
                    label.extend([f.iloc[i, 3]])
                    conf.extend([f.iloc[i, 4]])
                    idx += 1
        idx += 1

    data = {'video-id':videoid, 't-start':tstart, 't-end':tend, 'label':label, 'conf':conf}

    merged_predictions = pd.DataFrame.from_dict(data)
    # merged_predictions = merged_predictions[['video-id', 't-start', 't-end', 'label', 'conf']]
    # print("new f")
    # temp = merged_predictions[merged_predictions['video-id'] == '0291-L']
    # print("temp shape ", temp.shape)

    pickle.dump(merged_predictions, open('merged_predictions.pkl', 'wb'))

    return merged_predictions


def ap(lst, ratio, ground):
    lst.sort(key=lambda x: x[3])  # lst = sorted(lst, key=lambda x: x[3], reverse=True)  # sorted by confidence
    # print("lst")
    # print(lst)
    cos_map, count_map, positive = match(lst, ratio, ground)
    score = 0
    number_proposal = len(lst)
    number_ground = len(ground)
    old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)

    for x in range(len(lst)):
        number_proposal -= 1
        if (cos_map[x] == -1): continue
        count_map[cos_map[x]] -= 1
        if (count_map[cos_map[x]] == 0): positive -= 1

        precision, recall = calc_pr(positive, number_proposal, number_ground)
        if precision > old_precision:
            old_precision = precision
        score += old_precision * (old_recall - recall)
        old_recall = recall
    return score

def match(lst, ratio, ground):
    def overlap(prop, ground):
        l_p, s_p, e_p, c_p, v_p = prop
        l_g, s_g, e_g, c_g, v_g = ground
        if (int(l_p) != int(l_g)): return 0
        if (v_p != v_g): return 0
        return (min(e_p, e_g) - max(s_p, s_g)) / (max(e_p, e_g) - min(s_p, s_g))

    cos_map = [-1 for x in range(len(lst))]
    count_map = [0 for x in range(len(ground))]
    # generate index_map to speed up
    index_map = [[] for x in range(number_label)]
    for x in range(len(ground)):
        index_map[int(ground[x][0])].append(x)

    for x in range(len(lst)):
        for y in index_map[int(lst[x][0])]:
            if (overlap(lst[x], ground[y]) < ratio): continue
            if (overlap(lst[x], ground[y]) < overlap(lst[x], ground[cos_map[x]])): continue
            cos_map[x] = y
        if (cos_map[x] != -1): count_map[cos_map[x]] += 1
    positive = sum([(x > 0) for x in count_map])
    return cos_map, count_map, positive

def calc_pr(positive, proposal, ground):
    if (proposal == 0): return 0, 0
    if (ground == 0): return 0, 0
    return (1.0 * positive) / proposal, (1.0 * positive) / ground

def plot_fig(lst, ratio, ground, method):
    lst.sort(key=lambda x: x[3])  # sorted by confidence
    cos_map, count_map, positive = match(lst, ratio, ground)
    number_proposal = len(lst)
    number_ground = len(ground)
    old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)

    recalls = [old_recall]
    precisions = [old_precision]
    for x in range(len(lst)):
        number_proposal -= 1;
        if (cos_map[x] == -1): continue
        count_map[cos_map[x]] -= 1;
        if (count_map[cos_map[x]] == 0): positive -= 1;

        precision, recall = calc_pr(positive, number_proposal, number_ground)
        if precision > old_precision:
            old_precision = precision
        recalls.append(recall)
        precisions.append(old_precision)
        old_recall = recall
    fig = plt.figure()
    plt.axis([0, 1, 0, 1])
    plt.plot(recalls, precisions, 'r')
    plt.savefig('%s%s.png' % (fig_folder, method))

# from scipy.ndimage.filters import uniform_filter1d
from copy import deepcopy

def apply_mean_filter(cls_probs,k=0):
    length = cls_probs.shape[0]
    print("Number of cls_probs entries ", length)
    new_cls_probs = deepcopy(cls_probs)

    if k == 0:
        print("Mean filter not applied!")
    elif k == 3:
        print("Mean filter with k == 3 is applied!")
        for i in range(1, cls_probs.shape[0]-1):
            new_cls_probs[i] = np.average(np.vstack((cls_probs[i-1, :], cls_probs[i, :], cls_probs[i + 1, :])), axis=0)
    elif k == 5:
        print("Mean filter with k == 5 is applied!")
        for i in range(2, cls_probs.shape[0]-2):
            new_cls_probs[i] = np.average(np.vstack((cls_probs[i-2, :], cls_probs[i-1, :], cls_probs[i, :], cls_probs[i + 1, :], cls_probs[i+2, :])), axis=0)
    else:
        print("Invalid mean filter size specified!")
    """
    for i in range(1, cls_probs.shape[0]-1):
        #print(cls_probs[i-1, :])
        #print(cls_probs[i, :])
        #print(cls_probs[i+1, :])
        new_cls_probs[i] = np.average(np.vstack((cls_probs[i-1, :], cls_probs[i, :], cls_probs[i + 1, :])), axis=0)
        # new_cls_probs[i] = np.average(np.vstack((cls_probs[i-2, :], cls_probs[i-1, :], cls_probs[i, :], cls_probs[i + 1, :], cls_probs[i+2, :])), axis=0)
        #print("new cls probs ")
        #print(cls_probs[i])
        #break
    """
    return new_cls_probs

def get_prediction_res(idx, pred):
    pred = np.array(pred)
    res = pred[[idx]]
    for i in res:
        print(i)
    vid = [i[4] for i in res]
    lbl = [i[0] for i in res]
    start = [i[1] for i in res]
    end = [i[2] for i in res]
    score = [i[3] for i in res]

    data = {'video-id':vid, 't-start':start, 't-end':end, 'label':lbl, 'conf':score}
    df = pd.DataFrame.from_dict(data)
    df = df[['video-id','t-start','t-end','label','conf']]
    return df

if __name__ == "__main__":
    path = "../output"

    # below split for overlapping segments
    # test_split = pickle.load(open(os.path.join(path, 'test_split.pkl'), 'rb'))

    # below cls_probs for the corresponding overlapping segments
    # cls_probs = pickle.load(open(os.path.join(path, 'PKU_cls_probs.pkl'), 'rb'))

    # apply mean filter to the class probabilities
    # new_cls_probs = apply_mean_filter(cls_probs)

    # below split for overlapping segments
    # test_split = pickle.load(open(os.path.join(path, 'test_split_no_overlap.pkl'), 'rb'))
    # test_split = test_split[:9864]

    # Need to change below for different stacksize (64, 32, 16) ********************************
    # test_split = pickle.load(open(os.path.join(path, 'test_split_no_overlap_lstm.pkl'), 'rb'))  # for 64 stack size
    # print("length is ", len(test_split))

    # Validation videos used for 32 frames
    # test_split = pickle.load(open(os.path.join(path, 'validation_split_32frames.pkl'), 'rb'))  # for 32 stack size
    # test_split = test_split[:-10]  # to account for correct number of test splits  19802 actual to 19792 predicted
    # print("length is ", len(test_split))

    # only for 16 as number of frames
    test_split = pickle.load(open(os.path.join(path, 'validation_split_16frames.pkl'), 'rb'))  # for 16 stack size
    test_split = test_split[:-1]
    print("length is ", len(test_split))

    # only for 10 as number of frames
    # test_split = pickle.load(open(os.path.join(path, 'validation_split_10frames.pkl'), 'rb'))  # for 10 stack size
    # test_split = test_split[:-11]
    # print("length is ", len(test_split))

    # iterate through the proposals and create new prediction file taking care of overlapping 20 frames..
    validation_videos = ['0291-L', '0291-M', '0291-R', '0292-L', '0292-M', '0292-R', '0293-L', '0293-M', '0293-R',
                         '0294-L', '0294-M', '0294-R', '0295-L', '0295-M', '0295-R', '0296-L', '0296-M', '0296-R',
                         '0297-L', '0297-M', '0297-R', '0298-L', '0298-M', '0298-R', '0299-L', '0299-M', '0299-R',
                         '0300-L', '0300-M', '0300-R', '0301-L', '0301-M', '0301-R', '0302-L', '0302-M', '0302-R',
                         '0303-L', '0303-M', '0303-R', '0304-L', '0304-M', '0304-R', '0305-L', '0305-M', '0305-R',
                         '0306-L', '0306-M', '0306-R', '0307-L', '0307-M', '0307-R', '0308-L', '0308-M', '0308-R',
                         '0309-L', '0309-M', '0309-R', '0310-L', '0310-M', '0310-R', '0311-L', '0311-M', '0311-R',
                         '0312-L', '0312-M', '0312-R', '0313-L', '0313-M', '0313-R', '0314-L', '0314-M', '0314-R',
                         '0315-L', '0315-M', '0315-R', '0316-L', '0316-M', '0316-R', '0317-L', '0317-M', '0317-R',
                         '0318-L', '0318-M', '0318-R', '0319-L', '0319-M', '0319-R', '0320-L', '0320-M', '0320-R',
                         '0321-L', '0321-M', '0321-R', '0322-L', '0322-M', '0322-R', '0323-L', '0323-M', '0323-R',
                         '0324-L', '0324-M', '0324-R', '0325-L', '0325-M', '0325-R', '0326-L', '0326-M', '0326-R',
                         '0327-L', '0327-M', '0327-R', '0328-L', '0328-M', '0328-R', '0329-L', '0329-M', '0329-R',
                         '0330-L', '0330-M', '0330-R', '0331-L', '0331-M', '0331-R', '0332-L', '0332-M', '0332-R',
                         '0333-L', '0333-M', '0333-R', '0334-L', '0334-M', '0334-R']

    # align cls_probs obtained from GRU ..
    # filename = "PKU_GRU_features_test_by_video11.pkl" 70.07 mAP
    # filename = "PKU_GRU_clsprobs_test_by_video_epoch11.pkl" 72.62 mAP
    # filename = "PKU_GRU_clsprobs_test_by_video_epoch11_1024.pkl"  # 73.05 mAP using mean of 7 x 1024
    # filename = "PKU_GRU_clsprobs_test_by_video_epoch11_1024_max.pkl"  # 72.07 mAP using max of 7 x 1024
    # filename = "PKU_GRU_clsprobs_test_by_video_epoch11_1024_32.pkl"  # 80.26 mAP
    # filename = "PKU_GRU_clsprobs_test_by_video_epoch11_1024_LSTM_32.pkl"  # 81.71 mAP
    # filename = "PKU_GRU_clsprobs_test_by_video_epoch11_1024_LSTM_16.pkl"  # 82.55 mAP
    # filename = 'PKU_GRU_clsprobs_test_by_video_epoch11_1024_poseattnet_LSTM_16_epoch3_I3D.pkl'  # 82.36 mAP
    filename = 'PKU_GRU_clsprobs_1024_poseattnet_nopretrainedNTU_epoch5_test.pkl'  # 88.98 mAP
    # filename = 'PKU_GRU_clsprobs_1024_poseattnet_pretrainedNTU_epoch4.pkl'  # 85.31 mAP

    # filename = "PKU_LSTM_clsprobs_1024_I3D_feature_10frames_epoch6_test.pkl"  # for 10 frames I3D baseline check


    # filename = 'PKU_LSTM_clsprobs_1024_poseattnet_feature_10frames_epoch8_test.pkl'  #   88.38 mAP  10 frames evaluation
    # filename = 'PKU_LSTM_clsprobs_1024_poseattnet_feature_10frames_epoch3_test.pkl'  #   87.23 mAP  10 frames evaluation

    # Comparison of action detection methods on PKU-MMD for different combination of hyper-parameters value.
    cls_probs_dict = pickle.load(open(os.path.join(path, filename), 'rb'))
    cls_probs = []
    for i in validation_videos:
        probs = cls_probs_dict[i][0]
        # print("shape ", probs.shape)
        cls_probs.extend(probs)
    cls_probs = np.array(cls_probs)

    print("new cls probs shape ", cls_probs.shape)
    # below cls_probs for the corresponding overlapping segments
    # cls_probs = pickle.load(open(os.path.join(path, 'PKU_cls_probs_no_overlap_test.pkl'), 'rb'))
    # cls_probs = pickle.load(open(os.path.join(path, 'cls_probs_GRU.pkl'), 'rb'))

    # mAP reduced to 60 with k=3 and to 40 with k=5 if mean filter is applied
    # specify k as the window size of the mean filter to be applied! ******************
    cls_probs = apply_mean_filter(cls_probs, k=3) # value for best performance is k=3

    print("test_split shape ", len(test_split))
    print("cls probs shape ", cls_probs.shape)
    assert len(test_split) == cls_probs.shape[0]

    # predictions already saved as a pickle file . just load the pickle file to get predictions data
    prepare_prediction(test_split, name='no_overlap_')   # name = ''
    print("prepare prediction done ")

    # ground_truth = pickle.load(open(os.path.join(path, 'ground_truth_df.pkl'), 'rb'))
    ground_truth = pickle.load(open(os.path.join(path, 'ground_truth_dfnew.pkl'), 'rb'))
    predictions = pickle.load(open(os.path.join('predictions_no_overlap_df.pkl'), 'rb'))      # 'predictions_df.pkl'

    print("Number of ground_truth entries ", ground_truth.shape[0])
    print("Number of prediction entries ", predictions.shape[0])

    print("predictions")
    print(predictions[predictions['video-id'] == '0310-M'])

    # print("ground_truth")
    # print(ground_truth[0:30])

    # print("predictions")
    # print(predictions[0:20])

    # print("cls prob")
    # print(cls_probs.shape)
    # print(cls_probs[0])
    # print(np.argmax(cls_probs, axis=1)[0:20])

    # predictions['conf'] = 0

    # print("predictions shape  ", predictions.shape)
    # print("predictions before start ", predictions)

    # prepare the predictions as activity vs no activity (1 vs 0 at label)
    count = 0
    gamma = 0.4   # value for best performance 0.4

    """
    for i in range(predictions.shape[0]):
        tmp = sum(cls_probs[i, 1:])
        if cls_probs[i, 0] > tmp or tmp < gamma:
            predictions.iloc[i, 3] = 0
            predictions.iloc[i, 4] = cls_probs[i, 0]
        else:
            label = np.argmax(cls_probs[i, 1:])
            predictions.iloc[i, 3] = label
            predictions.iloc[i, 4] = cls_probs[i, 1:][label]

    """
    # print("predictions ******************************")
    t = predictions[predictions['video-id'] == '0291-L']
    # print(t)
    overlap = 20

    # creates segments which are non overlapping for next step of merging
    # new_prediction = get_distinct_segments(new_cls_probs, predictions)

    # below are from the segments which are already non overlapping ..
    new_prediction = get_distinct_segments_no_overlap(cls_probs, predictions)
    print("new predictions received as well")

    # print("new_predictions ")
    # print(new_prediction)
    # print("new predictions shape ", new_prediction.shape)

    # apply_multiscale_processing()

    split_prediction = pickle.dump(new_prediction, open('split_predictions.pkl', 'wb'))  # split_predictions

    split_prediction = pickle.load(open('split_predictions.pkl', 'rb'))

    # print("split prediction ")
    # print(split_prediction)

    # merge predictions from small segments into large sorted ...
    merge_predictions = get_merge_predictions(validation_videos, split_prediction)

    predictions_with_bkg = pickle.load(open('merged_predictions.pkl', 'rb'))

    # output commands to display the confidence scores of the predictions
    # print("predictions with bkg")
    # print(predictions_with_bkg)

    predictions_with_bkg = predictions_with_bkg[['label', 't-start', 't-end', 'conf', 'video-id']]

    # filter out background labels and prepare data for calculation fo mAP

    # is below needed for calculation of mAP ???
    # final_predictions = predictions_with_bkg[predictions_with_bkg['label'] != 0]

    final_predictions = predictions_with_bkg

    # print("unique labels in final_predictions ")
    # print(final_predictions['label'].unique())

    # print("unique labels in gt")
    # print(ground_truth['label'].unique())

    # print("ground_truth entries ")
    # print(ground_truth)

    # print("final prediction")
    # print(final_predictions[final_predictions['video-id'] == '0291-L'])

    # rearrange ground truth for calculation
    ground_truth = ground_truth[['label', 't-start', 't-end', 'conf', 'video-id']]

    number_label = 52

    v_props = []
    v_grounds = []

    for i in validation_videos:
        datap = final_predictions[final_predictions['video-id'] == i]
        datag = ground_truth[ground_truth['video-id'] == i]
        tmp = []
        for idx in range(datap.shape[0]):
            tmp.append([datap.iloc[idx, 0], datap.iloc[idx, 1], datap.iloc[idx, 2], datap.iloc[idx, 3], datap.iloc[idx, 4]])
        v_props.append(tmp)
        tmp1 = []
        for idx in range(datag.shape[0]):
            tmp1.append([datag.iloc[idx, 0], datag.iloc[idx, 1], datag.iloc[idx, 2], datag.iloc[idx, 3], datag.iloc[idx, 4]])
        v_grounds.append(tmp1)

    a_props = [[] for x in range(number_label)]
    a_grounds = [[] for x in range(number_label)]

    # print("v_props ")
    # print(v_props[85])
    # print(len(v_props))
    # print("vProps shep ", len(v_props))
    # print("v_ground ")
    # print(v_grounds)
    # print(len(v_grounds))

    for x in range(len(v_props)):
        for y in range(len(v_props[x])):
            a_props[int(v_props[x][y][0])].append(v_props[x][y])

    for x in range(len(v_grounds)):
        for y in range(len(v_grounds[x])):
            a_grounds[int(v_grounds[x][y][0])].append(v_grounds[x][y])

    # ========== find all proposals========
    all_props = sum(a_props, [])
    all_grounds = sum(a_grounds, [])

    theta = 0.1
    print("Threshold value for temporal IoU is "+str(theta))
    AP = ap(all_props, theta, all_grounds)
    # ap = ap(final_predictions.values, ratio, ground_truth.values)
    print("ap is ", AP)

    mAP = sum([ap(a_props[x + 1], theta, a_grounds[x + 1]) for x in range(number_label - 1)]) / (number_label - 1)
    print("mAP by action ", mAP)

    t = [ap(a_props[x + 1], theta, a_grounds[x + 1]) for x in range(number_label - 1)]
    print("mAP by class")
    print(t)

    mAP_per_video = [ap(v_props[x], theta, v_grounds[x]) for x in range(len(v_props))]  # / len(v_props)
    mAP_video = sum([ap(v_props[x], theta, v_grounds[x]) for x in range(len(v_props))]) / len(v_props)
    print("mAP video ", mAP_video)

    idx = [85, 58, 75]
    # top3pred = get_prediction_res(idx, v_props)
    # pickle.dump(top3pred, open('top3pred.pkl', 'wb'))
    # specify the name of the file to be saved for the video based mAP
    # pickle.dump(mAP_per_video, open('map_per_video_onlyI3D.pkl', 'wb'))
