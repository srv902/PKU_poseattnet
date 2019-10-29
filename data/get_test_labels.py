import os
import pandas as pd
import pickle

validation_videos = ['0291-L', '0291-M', '0291-R', '0292-L', '0292-M', '0292-R', '0293-L', '0293-M', '0293-R', '0294-L', '0294-M', '0294-R', '0295-L', '0295-M', '0295-R', '0296-L', '0296-M', '0296-R', '0297-L', '0297-M', '0297-R', '0298-L', '0298-M', '0298-R', '0299-L', '0299-M', '0299-R', '0300-L', '0300-M', '0300-R', '0301-L', '0301-M', '0301-R', '0302-L', '0302-M', '0302-R', '0303-L', '0303-M', '0303-R', '0304-L', '0304-M', '0304-R', '0305-L', '0305-M', '0305-R', '0306-L', '0306-M', '0306-R', '0307-L', '0307-M', '0307-R', '0308-L', '0308-M', '0308-R', '0309-L', '0309-M', '0309-R', '0310-L', '0310-M', '0310-R', '0311-L', '0311-M', '0311-R', '0312-L', '0312-M', '0312-R', '0313-L', '0313-M', '0313-R', '0314-L', '0314-M', '0314-R', '0315-L', '0315-M', '0315-R', '0316-L', '0316-M', '0316-R', '0317-L', '0317-M', '0317-R', '0318-L', '0318-M', '0318-R', '0319-L', '0319-M', '0319-R', '0320-L', '0320-M', '0320-R', '0321-L', '0321-M', '0321-R', '0322-L', '0322-M', '0322-R', '0323-L', '0323-M', '0323-R', '0324-L', '0324-M', '0324-R', '0325-L', '0325-M', '0325-R', '0326-L', '0326-M', '0326-R', '0327-L', '0327-M', '0327-R', '0328-L', '0328-M', '0328-R', '0329-L', '0329-M', '0329-R', '0330-L', '0330-M', '0330-R', '0331-L', '0331-M', '0331-R', '0332-L', '0332-M', '0332-R', '0333-L', '0333-M', '0333-R', '0334-L', '0334-M', '0334-R']
val_path = "/data/stars/user/rriverag/pku-mmd/PKUMMD/Label/Train_Label_PKU_final"

ground_truth = pd.DataFrame(columns=['video-id', 't-start', 't-end', 'label', 'conf'])
for f in validation_videos:
    labels = pd.read_csv(os.path.join(val_path,f+'.txt'), header=None)
    for i in range(labels.shape[0]):
        tmp = {'video-id':f, 't-start':labels.iloc[i,1], 't-end':labels.iloc[i,2], 'label':labels.iloc[i,0], 'conf':labels.iloc[i,3]}
        ground_truth = ground_truth.append(tmp, ignore_index=True)


pickle.dump(ground_truth, open('ground_truth_dfnew.pkl', 'wb'))

