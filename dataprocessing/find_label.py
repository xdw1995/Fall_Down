import os
import pandas as pd
import numpy as np
import re
mode = 'train'

label_path = '/data/xudw/labels'
p = '/data/xudw/Fall_Down_data'
f = open("now_data_"+mode+".txt",'w')
def is_odd(n):
    return n!= ' '
for SubjectaActivitybTrialCCamerasuibian in sorted(os.listdir(p)):

    s = SubjectaActivitybTrialCCamerasuibian
    c = re.findall('(\d+)',s)
    Subject = c[0]
    print(Subject)
    if mode=='test':
        if int(Subject) not in [13,14,15,16,17]:
            continue
    if mode=='train':
        if int(Subject) in [13,14,15,16,17]:
            continue
    Activity = c[1]
    Trial = c[2]
    # print(c)
    csv = label_path+'/'+'Subject'+str(Subject)+'/'+'Activity'+str(Activity)+'/'+'Subject'+str(Subject)+'Activity'+str(Activity)+'Trial'+str(Trial)+'.csv'

    csv_data = pd.read_csv(csv)

    loc = np.array(csv_data["Timestamp"])
    label = np.array(csv_data['Tag'])
    a = []
    b = []
    for index,pic in enumerate(sorted(os.listdir(p+'/'+SubjectaActivitybTrialCCamerasuibian))):
        if len(b)==5:
            f.write(''.join(filter(is_odd,list(str(a))))+' '+''.join(filter(is_odd,list(str(b)))))
            f.write('\n')
            a = []
            b = []
        try:
            int(label[index])
            if int(label[index])==7 or int(label[index])==11 or int(label[index])==6 or int(label[index])==8 or int(label[index])==10:
                a.append('/data/xudw/Fall_Down_data'+'/'+SubjectaActivitybTrialCCamerasuibian+'/'+pic)
                b.append(label[index])
        except:
            a = []
            b = []