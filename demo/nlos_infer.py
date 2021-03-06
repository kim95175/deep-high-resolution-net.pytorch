####### Image data를 읽어와 GT 를 만드는 코드 #######

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys
sys.path.append("../lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
from utils.vis import save_batch_heatmaps

import glob
from PIL import Image
import imageio

#CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
GPU_NUM = 0 # 원하는 GPU 번호 입력
CTX = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(CTX)
print(CTX)
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# 사진에서 먼저 사람의 위치를 추정.
def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes

# 예측한 keypoint의 좌표가 detection box 내에 있는지를 체크
def in_box(x, y, box):
    if not ((box[0][0] <= x) and (x <= box[1][0])):
        return False
    if not ((box[0][1] <= y) and (y <= box[1][1])):
        return False
    return True

# image로 부터 각 Keypoint의 좌표를 계산.
def get_pose_estimation_prediction(pose_model, image, centers, scales, box, transform):
    rotation = 0

    #print("img shape ", image.shape)
    #print("centers ", centers)
    #print("scales ", scales)
    #print(box)
    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        #print("trans", trans)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        img = model_input
        cv2.imwrite('trans_input.jpg', img)
        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)
    #zero_heatmap = np.zeros((120, 120), dtype=np.float32)
    zero_heatmap = torch.cuda.FloatTensor(120, 120).fill_(0)
    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    # heatmap output :
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))


    for idx1, mat in enumerate(coords[0]):
        x_coord, y_coord = int(mat[0]), int(mat[1])
        if not(in_box(x_coord, y_coord, box)):
            coords[0][idx1] = [-1, -1]
            output[0][idx1] = zero_heatmap


    return output, coords


# deteciton 결과를 기준으로 center, scale 계산
def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='../data/nlos/nlos_result/')
    #parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


# 좌표를 받아 heatmap 생성.
def generate_target(joints):
    '''
    :param joints:  [num_joints, 2]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = joints.shape[0]
    sigma = 2
    #print("num_joints ", num_joints)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    for idx, coord in enumerate(joints):
        #print("{} = {}".format(idx, coord))
        if(coord[0]== -1):
            target_weight[idx] = 0
        #print(target_weight[idx])

    heatmap_size = (120, 120)
    image_size = (480, 480)
    if True:
        target = np.zeros((num_joints,
                            heatmap_size[1],
                            heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            if target_weight[joint_id] ==0:
                continue
            feat_stride = (image_size[0] / heatmap_size[0], image_size[1] / heatmap_size[1] )
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            '''
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                       or br[0] < 0 or br[1] < 0:
                 # If not, just return the image as is
                target_weight[joint_id] = 0
                continue
            '''
            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    #if self.use_different_joints_weight:
    #    target_weight = np.multiply(target_weight, self.joints_weight)

    return target, target_weight



def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    #pose_dir = prepare_output_dirs(args.outputDir)

    # Deection model
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    # 학습된 모델 불러오기
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # img 파일 및 gt 저장할 폴더 불러오기

    #path_dir = '../..data/nlos/save_data_original/'
    path_dir = '/data/nlos/save_data_ver2/'
    path2 = os.listdir(path_dir)
    dirs = []
    img_dirs = []
    gt_dirs = []
    for dir in path2:
        dir_name = os.path.join(path_dir, dir)
        if os.path.isdir(dir_name):
            dir_name = dir_name
            dirs.append(dir_name)
            img_dirs.append(os.path.join(dir_name, 'img'))
            gt_dirs.append(os.path.join(dir_name, 'gt'))

    #print(dirs)
    #print(img_dirs)
    #print(gt_dirs)
    print("============= all {} dirs =============".format(len(dirs)))

    get_box = True

    for i in range(len(dirs)):
    #for i in [0]:
        input_dir = img_dirs[i]  
        output_dir = gt_dirs[i]
        print("input_dir ", input_dir)
        print("output_dir ", output_dir)

        input_file_list = os.listdir(input_dir)
        num_done = 0
        det_cnt = 0
        images= []
        for f in input_file_list:
            file_num = f[:5]
            if num_done%1000 == 0:
                print("{} images done".format(num_done))
            num_done += 1
            input_file = os.path.join(input_dir,f)

            img = cv2.imread(input_file)        # img 파일 불러와서 크기 조절하기
            img = cv2.resize(
                img,
                (480, 480),  # (width, height)
                interpolation=cv2.INTER_CUBIC
            )

            # print(img.shape)
            # detection model을 통해 먼저 사람이 있는 위치를 추정 ( BOX ).
            if get_box:
                detection_boxes = get_person_detection_boxes(box_model, img, threshold=0.9)
                full_boxes = []
                full_boxes.append([(0, 0), (480, 480)])
            #print(detection_boxes)
            centers = []
            scales = []

            # 사람의 위치가 detection 되지 않을 시 사진 전체를 BOX로 함.
            if not detection_boxes:
                box = full_boxes[0]
            else:
                det_cnt = det_cnt + 1
                box = detection_boxes[0]


            # box를 image위에 나타내고, box 를 기준으로 center, scale 계산 -> center, scale을 기준으로 image를 조정하여 학습 성능을 높임.
            cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

            # image로부터 신체 각 Keypoint의 좌표를 계산함.
            output, pose_preds = get_pose_estimation_prediction(pose_model, img, centers, scales, box, transform=pose_transform)
            #output_np = output.cpu().detach().numpy()
            #output_np = np.concatenate((output_np[0][0, :, :].reshape(1, 120, 120), output_np[0][5:, :, :]), axis=0)
            
            # 얼굴에 대한 정보 (눈, 귀) 에 대한 정보 제외.
            tmp_preds = np.concatenate((pose_preds[0][0].reshape(1, 2), pose_preds[0][5:]))
            pose_preds = tmp_preds.reshape(1, 13, 2)

            # 좌표를 기반으로 heatmap 생성
            hm, _ = generate_target(pose_preds[0])
            
            
            #print("hm {}".format(hm.shape))
            save_dir = output_dir + "/" + file_num
            np.save(save_dir, hm)  # heatmap 저장.
            
            for idx1, mat in enumerate(pose_preds[0]):
                x_coord, y_coord = int(mat[0]), int(mat[1])
                if idx1 == 0:
                    cv2.circle(img, (x_coord, y_coord), 3, (0, 0, 255), -1)
                elif idx1 in [1, 3, 5]:  # green
                    cv2.circle(img, (x_coord, y_coord), 3, (0, 255, 0), -1)
                elif idx1 in [2, 4, 6]:  # blue
                    cv2.circle(img, (x_coord, y_coord), 3, (255, 0, 0), -1)
                elif idx1 in [7, 9, 11]:  # 청록
                    cv2.circle(img, (x_coord, y_coord), 3, (255, 255, 255), -1)
                elif idx1 in [8, 10, 12]:  # yello
                    cv2.circle(img, (x_coord, y_coord), 3, (0, 255, 255), -1)

            #images.append(img)  

            '''
            trans_img = pose_transform(img)
            trans_img = trans_img.reshape(1, 3, 480, 480)
            torch_hm = torch.from_numpy(hm)
            torch_hm = torch_hm.reshape(1, 13, 120, 120)
            save_batch_heatmaps(
                trans_img, torch_hm, 'my_pred.jpg'
                # model_inputs, tmp, 'test_hm_pred.jpg'
            )
            if num_done ==100:
                break #for test only once
            '''
        print("detection : fail {}, success {}".format(num_done - det_cnt, det_cnt))
        #imageio.mimsave("test{}.gif".format(i), images, fps=18)

if __name__ == '__main__':
    main()
