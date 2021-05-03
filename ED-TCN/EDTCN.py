import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import os
from PIL import Image, ImageOps
import time
import glob
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
import math
import cv2
from torch.utils.data import Sampler
from torch.nn import functional as F
import sklearn.metrics as sklmetrics
from sklearn.preprocessing import normalize


parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-s', '--seq', default=16, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train', default=128, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val', default=128, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=100, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-4, type=float, help='learning rate for optimizer, default 1e-3')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=0, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')

parser.add_argument('--annotation', default='F:/Feature/MISAW/Procedural decription/annotation.txt', type=str, help='annotation path')
parser.add_argument('--video', default='F:/Feature/MISAW/Video/', type=str, help='Video path')
parser.add_argument('--image', default='F:/Feature/MISAW/Images/', type=str, help='Video to images save path')
parser.add_argument('--data', default='F:/Feature/MISAW/Kinematic/', type=str, help='Video to kdata save path')
parser.add_argument('--srate', default=5, type=int, help='sample rate')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

annotation = args.annotation
video = args.video
image = args.image
data = args.data
srate = args.srate
train_ratio = 0.7
val_ratio = 0.3

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.crop((0, 0, img.size[0] / 2, img.size[1]))
            return img.convert('RGB')


class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CholecDataset(Dataset):
    def __init__(self, image, file_paths, file_labels, kdata,
                 transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels
        self.image = image
        self.kdata = kdata
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = image + self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        data = self.kdata[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase, data

    def __len__(self):
        return len(self.file_paths)

# import torchvision
# model = torchvision.models.resnet50(pretrained=True)
class res34_tcn(torch.nn.Module):
    def __init__(self):
        super(res34_tcn, self).__init__()
        resnet = models.resnet34(pretrained=True)  # from torchvision import models
        self.share = torch.nn.Sequential() # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)  # 如果不这么做就要去除原resnet的全连接层
        self.tcn1 = TCN(5, 64, 512, 7)
        self.tcn2 = TCN(5, 64, 16, 7)
        self.conv_out_classes = nn.Conv1d(64, 7, 1)
        #self.tcn3 = MultiTCN(2, 5, 64, 512, 7)
        #self.tcn4 = MultiTCN(4, 5, 16, 16, 7)

    def forward(self, x, d):#
        x = self.share.forward(x)  # 特征
        x= x.view(-1, 512, sequence_length)
        y = self.tcn1(x)
        d = d.view(-1, 16, sequence_length)
        k = self.tcn2(d)
        c = torch.add(y, k)
        # c = torch.cat([y, k], dim=1)
        z = self.conv_out_classes(c)
        z =z.contiguous().view(-1, 7)
        #z = self.tcn3(z)

        return z


#########################################
class TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal_conv=False):
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.dim = dim
        self.num_classes = num_classes
        self.causal_conv = causal_conv
        super(TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(self.dim, self.num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, self.num_f_maps, self.num_f_maps, self.causal_conv))
             for i in range(self.num_layers)])
        #self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)  # [32, 64, 1]
        for layer in self.layers:
            out = layer(out)
        z = out
        return z


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, causal_conv=False, kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(dilation * (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))  # [128, 128, 3],[128, 128, 5],[128, 128, 9],[128, 128, 17],[128, 128, 33]
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)  # [128, 128, 1],[128, 128, 1],[128, 128, 1],[128, 128, 1],[128, 128, 1]
        out = self.dropout(out)  # [128, 128, 1],[128, 128, 1],[128, 128, 1],[128, 128, 1],[128, 128, 1]
        return (x + out)

##############################################


def get_useful_start_idx(sequence_length, list_each_length):
    # list_each_length是每个视频的帧数组成的集合
    count = 0
    idx = []
    for i in range(len(list_each_length)):  #10
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length * srate), sequence_length * srate):  # start ID在最后几个构不成一个batch
            idx.append(j)
        count += list_each_length[i]
    return idx

def frame_count(video, video_path):
    frames = []
    for v in video_path:
        mp4 = cv2.VideoCapture(video + v)  # 读取视频. 例如：F:/Feature/MISAW/Video/1_1.mp4
        frame_count = mp4.get(7)  # 视频文件中的帧数
        frames.append(int(frame_count))  # 每个视频的帧数组成的数组
        #frames.append(int(np.ceil(frame_count/25)))
    return frames


def get_data():
    '''
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    '''

    video_path = os.listdir(video)
    #print(video_path) #'1_1.mp4', '1_2.mp4', ...
    test_path = video_path[13:23]
    train_path1 = video_path[0:13]
    train_path2 = video_path[23:]
    train_path = train_path1+train_path2

    '''list = glob.glob(os.path.join(video, "*.mp4*"))
    for ID in list:
        GT_Id = ID[ID.rfind('\\') + 1:ID.rfind('_')]  ##Windows
        print(GT_Id)'''
    anno = open(annotation, 'r')
    image_path = os.listdir(image)
    labels_path = anno.readlines()
    frames = frame_count(video, video_path)
    # print(frames)  [7554, 5525, 6839, 9938, 9368, 11932, 14644, 7499, 12949, 5948, 4329, 4707, 5820, 3783, 7232, 6031, 4202]
    video_num = len(video_path)  # 17
    test_num = len(test_path)
    train_num = len(train_path)
    train_num_each1 = frame_count(video, train_path1)  # 0:10, train_ratio = 0.6
    train_num_each2 = frame_count(video, train_path2)
    train_num_each = train_num_each1 + train_num_each2
    # print(len(train_num_each)) 10
    val_num_each = frame_count(video, test_path)
    #test_num_each = frames[math.floor(video_num * (train_ratio + val_ratio)):]  # 13:
    train_num1 = sum(train_num_each1)
    train_num2 = sum(train_num_each2)
    # print(train_num) 92196
    val_num = sum(val_num_each)


    train_paths = image_path[0:train_num1] + image_path[train_num1+ val_num:]
    val_paths = image_path[train_num1:train_num1+ val_num]
    #test_paths = image_path[train_num + val_num:]
    labels_path = [l.split('\t')[1] for l in labels_path]  #####修改标号数字
    '''
    classes = labels_path
    classes = sorted(set(classes), key = classes.index)
    print(classes)
    '''
    #label_dict = {'Preparation\n': 0, 'CalotTriangleDissection\n': 1, 'ClippingCutting\n': 2,
                  #'GallbladderDissection\n': 3, 'GallbladderPackaging\n': 4, 'CleaningCoagulation\n': 5,
                  #'GallbladderRetraction\n': 6}
    #label_dict = {'Idle':0, 'Suturing':1, 'Knot tying':2}
    label_dict = {'Idle':0, 'Needle holding':1, 'Suture making':2, 'Suture handling':3, '1 knot':4, '2 knot':5, '3 knot':6, 'Idle Step':0}
    #label_dict = {'Idle': 0, 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'G5': 5,
                #  'G6': 6, 'G7': 7, 'G8': 8, 'G9': 9, 'G10': 10, 'G11': 11, 'G12': 12, 'G13': 13, 'G14': 14, 'G15': 15}
    labels = [label_dict[l] for l in labels_path]

    train_labels = labels[0:train_num1] + labels[train_num1+ val_num:]
    val_labels = labels[train_num1:train_num1 + val_num]
    #test_labels = labels[train_num + val_num:]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    #print('test_paths   : {:6d}'.format(len(test_paths)))
    #print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)  # 将数据类型转换为 np.int64
    val_labels = np.asarray(val_labels, dtype=np.int64)
    #test_labels = np.asarray(test_labels, dtype=np.int64)

    data_path = os.listdir(args.data)
    kdata = []
    for d in data_path:
        data = open(args.data + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        kdata = kdata + data

    kdata = np.asarray(kdata, dtype=np.float32)
    kdata = normalize(kdata, axis=0, norm='max')

    train_kdata = np.vstack((kdata[0:train_num1], kdata[train_num1+ val_num:]))
    val_kdata = kdata[train_num1:train_num1+ val_num]
    #test_kdata = kdata[train_num + val_num:, :]

    # torchvision.transforms是pytorch中的图像预处理包
    if use_flip == 0:  # not flip
        train_transforms = transforms.Compose([  # Compose把多个步骤整合到一起
            transforms.Resize([256, 256]),  # 把给定的图片resize到given size
            RandomCrop(224),
            transforms.ToTensor(),  # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])  # 用均值和标准差归一化张量图像
        ])
    elif use_flip == 1:  # flip
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224),
            RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    train_dataset = CholecDataset(image, train_paths, train_labels, train_kdata, train_transforms)
    val_dataset = CholecDataset(image, val_paths, val_labels, val_kdata, test_transforms)
    #test_dataset = CholecDataset(image, test_paths, test_labels, test_kdata, test_transforms)
    '''train_dataset = CholecDataset(image, train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(image, val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(image, test_paths, test_labels, test_transforms)'''

    return train_dataset, train_num_each, val_dataset, val_num_each


# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

########################################
'''
def compute_metrics(ground_truth, predicted_sequence):
    """
    :param ground_truth: real sequence
    :param predicted_sequence:  sequence predicted by network
    :param predicted_frame : list of frame predicted
    :return: different metric as accuracy and Average depend accuracy
    """
    downsample_ground_truth = ground_truth
    # extract all unique label from GT and predicted sequence
    all_label = np.concatenate((ground_truth, predicted_sequence), axis=0)
    #print(all_label)
    unique_label = list(set(all_label))
    unique_label.sort()



    performance = {}
    performance['accuracy'] = sklmetrics.balanced_accuracy_score(downsample_ground_truth, predicted_sequence)
    performance['precision'] = sklmetrics.precision_score(downsample_ground_truth, predicted_sequence, average='weighted', labels=unique_label)
    performance['recall'] = sklmetrics.recall_score(downsample_ground_truth, predicted_sequence, average='weighted', labels=unique_label)
    performance['f1'] = sklmetrics.f1_score(downsample_ground_truth, predicted_sequence, average='weighted', labels=unique_label)
    #performance['matrix'] = sklmetrics.confusion_matrix(downsample_ground_truth, predicted_sequence, labels=unique_label)
    return performance'''

######################################

def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    #print('train_useful_start_idx ',train_useful_start_idx )
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    #print('test_useful_start_idx ', val_useful_start_idx)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    # print('num_train_we_use',num_train_we_use) #92166
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # print('num_val_we_use', num_val_we_use)
    # num_train_we_use = 8000
    # num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]  # 训练数据开始位置
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    np.random.seed(0)
    np.random.shuffle(train_we_use_start_idx)  # 将序列的所有元素随机排序
    train_idx = []
    for i in range(num_train_we_use):  # 训练集帧数
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j * srate)  # 训练数据位置，每一张图是一个数据
    # print('train_idx',train_idx)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j * srate)
    # print('val_idx',val_idx)

    num_train_all = float(len(train_idx))
    num_val_all = float(len(val_idx))
    print('num of train dataset: {:6d}'.format(num_train))
    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(int(num_train_all)))
    print('num of valid dataset: {:6d}'.format(num_val))
    print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    print('num of valid we use : {:6d}'.format(num_val_we_use))
    print('num of all valid use: {:6d}'.format(int(num_val_all)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        # sampler=val_idx,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False
    )
    model = res34_tcn()
    if use_gpu:
        model = model.cuda()

    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # model.parameters()与model.state_dict()是Pytorch中用于查看网络参数的方法。前者多见于优化器的初始化,后者多见于模型的保存
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    record_np = np.zeros([epochs, 4])

    for epoch in range(epochs):
        np.random.seed(epoch)
        np.random.shuffle(train_we_use_start_idx)  # 将序列的所有元素随机排序
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j * srate)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=workers,
            pin_memory=False
        )

        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()
        num = 0
        train_num = 0
        for data in train_loader:
            num = num + 1
            # inputs, labels_phase = data
            inputs, labels_phase, kdata = data
            if use_gpu:
                inputs = Variable(inputs.cuda())  # Variable就是一个存放会变化值的地理位置，里面的值会不停发生变化
                labels = Variable(labels_phase.cuda())
                kdatas = Variable(kdata.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_phase)
                kdatas = Variable(kdata)
            optimizer.zero_grad()  # 梯度初始化为零,也就是把loss关于weight的导数变成0.
            # outputs = model.forward(inputs)  # 前向传播
            outputs = model.forward(inputs, kdatas)
            #outputs = F.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs.data, -1)  # .data 获取Variable的内部Tensor;torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引
            #_, yp = torch.max(y.data, 1)
            #print(yp)
            # print(yp.shape)
            print(num)
            print(preds)
            print(labels)


            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            train_corrects += torch.sum(preds == labels.data)
            train_num += labels.shape[0]
            print(train_corrects.cpu().numpy() / train_num)
            if train_corrects.cpu().numpy() / train_num > 0.75:
                torch.save(copy.deepcopy(model.state_dict()), 'test.pth')  # .state_dict()只保存网络中的参数(速度快，占内存少)

        train_elapsed_time = time.time() - train_start_time

        #train_accuracy1 = train_corrects1.cpu().numpy() / train_num
        train_accuracy = train_corrects.cpu().numpy() / train_num
        train_average_loss = train_loss / train_num

        # begin eval
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        val_start_time = time.time()
        for data in val_loader:
            inputs, labels_phase, kdata = data
            #inputs, labels_phase = data
            #labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
            #kdata = kdata[(sequence_length - 1)::sequence_length]
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_phase.cuda())
                kdatas = Variable(kdata.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_phase)
                kdatas = Variable(kdata)

            if crop_type == 0 or crop_type == 1:
                #outputs = model.forward(inputs)
                outputs = model.forward(inputs, kdatas)
            elif crop_type == 5:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs = model.forward(inputs, kdatas)
                # outputs = model.forward(inputs)
                outputs = outputs.view(5, -1, 3)
                outputs = torch.mean(outputs, 0)
            elif crop_type == 10:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs = model.forward(inputs, kdatas)
                #outputs = model.forward(inputs)
                outputs = outputs.view(10, -1, 3)
                outputs = torch.mean(outputs, 0)

            #outputs = outputs[sequence_length - 1::sequence_length]

            _, preds = torch.max(outputs.data, -1)
            #_, yp = torch.max(y.data, 1)
            print(num)
            print(preds)
            print(labels)


            loss = criterion(outputs, labels)
            #loss = 0.05 * loss1 + 0.15 * loss2 + 0.3 * loss3 + 0.5 * loss4
            #loss = 0.05 * loss1 + 0.1 * loss2 + 0.25 * loss3 + 0.6 * loss4
            val_loss += loss.data
            val_corrects += torch.sum(preds == labels.data)
            val_num += labels.shape[0]
        val_elapsed_time = time.time() - val_start_time
        val_accuracy = val_corrects.cpu().numpy() / val_num
        val_average_loss = val_loss / val_num
        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss: {:4.4f}'
              ' train accu: {:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid loss: {:4.4f}'
              ' valid accu: {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss,
                      train_accuracy,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss,
                      val_accuracy))

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        if val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())


        record_np[epoch, 0] = train_accuracy
        record_np[epoch, 1] = train_average_loss
        record_np[epoch, 2] = val_accuracy
        record_np[epoch, 3] = val_average_loss
        np.save(str(epoch) + '.npy', record_np)

    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))

    save_val = int("{:4.0f}".format(best_val_accuracy * 10000))
    save_train = int("{:4.0f}".format(correspond_train_acc * 10000))
    model_name = "tcn" \
                 + "_epoch_" + str(epochs) \
                 + "_length_" + str(sequence_length) \
                 + "_opt_" + str(optimizer_choice) \
                 + "_mulopt_" + str(multi_optim) \
                 + "_flip_" + str(use_flip) \
                 + "_crop_" + str(crop_type) \
                 + "_batch_" + str(train_batch_size) \
                 + "_train_" + str(save_train) \
                 + "_val_" + str(save_val) \
                 + ".pth"

    torch.save(best_model_wts, model_name)

    record_name = "tcn" \
                  + "_epoch_" + str(epochs) \
                  + "_length_" + str(sequence_length) \
                  + "_opt_" + str(optimizer_choice) \
                  + "_mulopt_" + str(multi_optim) \
                  + "_flip_" + str(use_flip) \
                  + "_crop_" + str(crop_type) \
                  + "_batch_" + str(train_batch_size) \
                  + "_train_" + str(save_train) \
                  + "_val_" + str(save_val) \
                  + ".npy"
    np.save(record_name, record_np)


def main():
    train_dataset, train_num_each, val_dataset, val_num_each= get_data()
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()