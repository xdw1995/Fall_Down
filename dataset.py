import os
from pathlib import Path
import re
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import sys
sys.path.append("../processing")
from zhongjichuli import video_transform,test_transform
class VideoDataset_single(Dataset):
    def __init__(self,mode='train'):
        # self.short_side = [128, 160]
        # self.crop_size = 112
        self.short_side = [256, 320]
        self.crop_size = 224
        self.mode = mode
        self.fname = []
        self.labels = []
        self.dic = {}
        self.dic[7]=0
        self.dic[11]=1
        self.dic[6]=2
        self.dic[8]=3
        self.dic[10]=4
        if mode=='train':
            p = '/data/xudw'
            p = p+'/'+'now_data_train.txt'
        if mode=='test':
            p = '/data/xudw'
            p = p+'/'+'now_data_test.txt'
        f = open(p,'r')
        for i in f.readlines():
            i = i.strip()

            list_path = eval(i.split(' ')[0])
            list_label = eval(i.split(' ')[1])
            self.fname.append(list_path[-1])
            self.labels.append(self.dic[int(list_label[-1])])
            # for i in range(5):
            #     self.fname.append(list_path[i])
            #     self.labels.append(self.dic[int(list_label[i])])

        # if self.mode =='test':
        #     path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/mod-ucf101-test.txt'
        #     with open(path,'r') as f:
        #         for i in f.readlines():
        #             for _ in range(10):#vote
        #                 self.fnames.append(video_path+'/'+i.split(' ')[0].strip())
                    # self.labels.append(int(i.split(" ")[1].strip())-1)
        # print(self.fnames)
        # print(self.labels)
        # print(self.fnames)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        frame = cv2.imread(self.fname[index])
        # frame = cv2.imread(self.fname[index])
        # a = re.findall(r'Camera(\d+)',self.fname[index])[0]
        # if a=='1':
        #     frame = cv2.imread(self.fname[index])
        #     frame = frame[172:480,59:488,:]
        #
        # if a=='2':
        #     frame = cv2.imread(self.fname[index])
        #     frame = frame[73:459,208:394,:]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            if np.random.random() < 0.5:
                frame = cv2.flip(frame, flipCode=1)

        #size =(112,112)
        #if size要改，crop函数里面的随机裁剪也要跟着改

        # print(buffer.shape)
        # ________________________________________debug
        # b = video_transform(frame)
        # b= b.numpy().transpose((1,2,3,0))
        #
        # for i in b:
        #     cv2.imshow(str(self.labels[index]),i)
        #     cv2.waitKey(1000)


        if self.mode == 'test':
            return test_transform(frame), torch.from_numpy(np.array(self.labels[index])).long()
        else:
            # print(video_transform(buffer).shape,"123")
            return video_transform(frame), torch.from_numpy(np.array(self.labels[index])).long()

    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer

    def __len__(self):
        return len(self.labels)


class VideoDataset_3_frames(Dataset):

    def __init__(self,mode='train'):
        # self.short_side = [128, 160]
        # self.crop_size = 112
        self.short_side = [256, 320]
        self.crop_size = 224
        self.mode = mode
        self.fname = []
        self.labels = []
        self.dic = {}
        self.dic[7]=0
        self.dic[11]=1
        self.dic[6]=2
        self.dic[8]=3
        self.dic[10]=4
        if mode=='train':
            p = '/data/xudw'
            p = p+'/'+'now_data_train.txt'
        if mode=='test':
            p = '/data/xudw'
            p = p+'/'+'now_data_test.txt'
        f = open(p,'r')
        for i in f.readlines():
            i = i.strip()

            list_path = eval(i.split(' ')[0])
            list_label = eval(i.split(' ')[1])

            # print(list_path)
            self.fname.append(list_path)
            self.labels.append(self.dic[int(list_label[-1])])


        # print(self.fnames)
        # print(self.labels)
        # print(self.fnames)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fname[index])

        if self.mode == 'train':
            buffer = self.randomflip(buffer)

        #size =(112,112)
        #if size要改，crop函数里面的随机裁剪也要跟着改
        temp = []
        for i in buffer:
            # cv2.imshow("1",i)
            # cv2.waitKey(10)
            temp.append(i)
        buffer = np.concatenate(temp, axis=2)
        # print(buffer.shape)
        # ________________________________________debug
        # b = video_transform(buffer)
        #
        # b= b.numpy().transpose((1,2,3,0))
        # for i in b:
        #     cv2.imshow("123",i)
        #     cv2.waitKey(100)

        if self.mode == 'test':
            return test_transform(buffer), torch.from_numpy(np.array(self.labels[index])).long()
        else:
            # print(video_transform(buffer).shape,"123")
            return video_transform(buffer), torch.from_numpy(np.array(self.labels[index])).long()

    def loadvideo(self, fname_list):
        buffer = []
        for i in fname_list:
            a = re.findall(r'Camera(\d+)',i)[0]
            if a=='1':
                frame = cv2.imread(i)
                frame = frame[172:480,59:488,:]

            if a=='2':
                frame = cv2.imread(i)
                frame = frame[73:459,208:394,:]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(frame)
        return np.array(buffer)

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    def __len__(self):
        return len(self.labels)
if __name__ == '__main__':

    datapath = '/disk/data/UCF-101'
    # print(len(VideoDataset( mode='validation',clip_len=5)))
    # print(len(VideoDataset( mode='train',clip_len=5)))
    train_dataloader = \
        DataLoader( VideoDataset_single(mode='train'), batch_size=32, shuffle=True, num_workers=0)
    # test = \
    #     DataLoader( VideoDataset( mode='validation'), batch_size=32, shuffle=False, num_workers=0)
    # print(len(train_dataloader))
    # print(len(test))

    for step, (buffer, label) in enumerate(train_dataloader):
        print(buffer.shape)
        print(label)
        # print("label: ", label)
    # test_dataloader = \
    #     DataLoader( VideoDataset_RDBdiff( mode='test',clip_len=64), batch_size=1, shuffle=False, num_workers=0)
    # for i in train_dataloader:
    #     print(i[0].shape)
    #     print(i[1])