from reid import ReID_feat
import os
import cv2
import torch
import matching
import random
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import os
from torchvision.transforms import functional as F
from PIL import Image
from collections import defaultdict
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
# image_encoder = ReID_feat.ImageEncoder('resnet50_ibn_a')#'reid/resnet50_ibn_a_model_120.pth'



def generate_boxfeature(image,boxes):
    features = image_encoder.encoder(image, boxes, (image.shape[1],image.shape[0]))
    features = normalize(features, axis=-1)
    return boxes, features

def iou(bb_test,bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return torch.min(dist)
def associate_detections_to_trackers(bbox_points, features, track_boxs, tracker_features, iou_threshold=0.3):
    result = bbox_points
    iou_matrix = np.zeros((len(bbox_points), len(track_boxs)), dtype=np.float32)
    # pose_matrix = np.zeros((len(bbox_points),len(track_boxs)),dtype=np.float32)
    feat_matrix = np.zeros((len(bbox_points), len(track_boxs)), dtype=np.float32)
    # dists = matching.iou_distance(bbox_points, track_boxs)

    for d in range(len(bbox_points)):
        for t in range(len(track_boxs)):
            # pose_matrix[d,t] = get_pose_matching_score(bbox_points[d], track_boxs[t], trackers[t])
            iou_matrix[d,t]  = iou(bbox_points[d],track_boxs[t])

            # feat_matrix[d,t] = euclidean_dist(features[d].unsqueeze(0), tracker_features[t].unsqueeze(0))

    dists = -iou_matrix
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
    for cur, pre in matches:
        # print(cur,pre)
        result[cur] = track_boxs[pre]
    return result

# a = associate_detections_to_trackers([[5,5,15,18],[1,1,12,12],[0,0,10,12]],1,[[5,5,15,18],[5,5,15,18]],1)
# print(a)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
fastrcnn = fasterrcnn_resnet50_fpn(True).cuda()
fastrcnn.eval()
# img = Image.open('/data/Dataset/HIE20/train/26/000458.jpg').convert("RGB")
# f = F.to_tensor(img)
# outputs = fastrcnn([f.cuda(),f.cuda()])
# box_xyxy = outputs[0]['boxes']
# mm = (box_xyxy[:,3]-box_xyxy[:,1])*(box_xyxy[:,2]-box_xyxy[:,0])*outputs[0]['scores']
# _, pred = mm.topk(max((1,3)), 0, True, True)
# now_box = box_xyxy[pred]
# print(now_box)
def produce_frames(framelist,model=fastrcnn,topk=1):
    # import pdb
    # pdb.set_trace()
    dic = defaultdict(list)
    buffer = []
    buffer_cv = []
    for path in framelist:
        img = Image.open(path).convert("RGB")
        f = F.to_tensor(img)
        if torch.cuda.is_available():
            f = f.cuda()
        buffer.append(f)
        buffer_cv.append(np.array(img))
    outputs = model(buffer)
    box_xyxy = outputs[0]['boxes']
    # print(box_xyxy)
    # print(123)
    # import pdb
    # pdb.set_trace()
    labels1 = outputs[0]['labels']==1
    labels2 = outputs[0]['scores']>0.7
    labels = labels1&labels2
    box_xyxy = box_xyxy[labels]

    scores = outputs[0]['scores']
    scores = scores[labels]
    # import pdb
    # pdb.set_trace()

    mm = (box_xyxy[:, 3] - box_xyxy[:, 1]) * (box_xyxy[:, 2] - box_xyxy[:, 0]) * scores/(2*torch.max(box_xyxy[:,0],box_xyxy[:,1]))
    # try:
    _, pred = mm.topk(max((1,topk)), 0, True, True)
    # except:
    #     print('121212121212')
    box_xyxy = box_xyxy.cpu().detach().numpy()
    now_box = box_xyxy[pred.cpu().detach().numpy()]
    # ffff= buffer_cv[0]
    # for index,ccc in enumerate(box_xyxy):
    #     ccc = list(map(int,ccc))
    #     cv2.rectangle(ffff,(ccc[0],ccc[1]),(ccc[2],ccc[3]),(0,255,0),2)
    # cv2.imwrite('xdw.jpg',ffff)
    # print(now_box)

    for index, box in enumerate(now_box):
        box = list(map(int,box))
        dic[index].append(cv2.resize(buffer_cv[0][box[1]:box[3],box[0]:box[2],:],(224,224)))
    # a = 0
    for other_frame in range(1,len(framelist)):
        otherboxes = associate_detections_to_trackers(now_box,1,outputs[other_frame]['boxes'].cpu().detach().numpy(),1)
        for index, box in enumerate(otherboxes):
            box = list(map(int, box))
            kkk = cv2.resize(
                buffer_cv[other_frame][box[1]:box[3], box[0]:box[2], :],
                (224,224)
            )
            dic[index].append(kkk)
            # cv2.imwrite(str(a)+'.jpg',kkk)
            # a = a+1
    return np.array(dic[0])
# print(produce_frames(['/data/xudw/Fall_Down_data/Subject10Activity3Trial2Camera2/','/data/Dataset/HIE20/train/26/000458.jpg','/data/Dataset/HIE20/train/26/000458.jpg']))
# a = []
# for i in sorted(os.listdir('/data/xudw/Fall_Down_data/Subject13Activity3Trial2Camera2')):
#     a.append('/data/xudw/Fall_Down_data/Subject13Activity3Trial2Camera2/'+i)

# d = produce_frames(a[40:42])
# print(d.shape)
# p = 1
# import re
# # import pdb
# # pdb.set_trace()
# import re
# num = 0
# for name in os.listdir('/data/xudw/Fall_Down_data'):
#     video = '/data/xudw/Fall_Down_data/'+name
#     a = re.findall(r'Subject(\d+)', name)[0]
#     # print(a)
#     if int(a) in [13,14,15,16,17]:
#         continue
#
#     for index,f in enumerate(sorted(os.listdir(video))):
#         fp = video+'/'+f
#         try:
#             data = produce_frames([fp])
#             aaa = 'A'.join(fp.split('/'))
#             print(aaa)
#             cv2.imwrite(aaa + '.jpg', data[0])
#             p = p + 1
#         except:
#             break
#         if index==0:
#             break
#     num = num+1
# print(num)
