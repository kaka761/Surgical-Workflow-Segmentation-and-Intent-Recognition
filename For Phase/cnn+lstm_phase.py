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
import pickle
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
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-s', '--seq', default=16, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train', default=128, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val', default=128, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=28, type=int, help='epochs to train and val, default 25')
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
    def __init__(self, image, file_paths, file_labels,
                 transform=None, loader=pil_loader):# kdata,
        self.file_paths = file_paths
        self.file_labels_phase = file_labels
        #self.kdata = kdata
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = image + self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        #data = self.kdata[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase#, data

    def __len__(self):
        return len(self.file_paths)

 # # import torchvision
# model = torchvision.models.resnet50(pretrained=True)
class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()  # 继承ResNet网络结构
        resnet = models.resnet34(pretrained=True)  # from torchvision import models
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)  # 继承ResNet网络结构
        self.lstm = nn.LSTM(512, 128, batch_first=True)  # 新增一个LSTM层
        #self.lstm2 = nn.LSTM(16, 128, batch_first=True)
        self.fc = nn.Linear(128, 3)   # 将原来的fc层改成fclass层,原来的fc层：self.fc = nn.Linear(512 * block.expansion, num_classes)。####注意修改类别

        init.xavier_normal_(self.lstm.all_weights[0][0])  # 没有预训练，则使用xavier初始化
        init.xavier_normal_(self.lstm.all_weights[0][1])
        #init.xavier_normal_(self.lstm2.all_weights[0][0])
        #init.xavier_normal_(self.lstm2.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x, d):  # forward方法中主要是定义数据在层之间的流动顺序，也就是层的连接顺序
        x = self.share.forward(x)  # 原来的x = self.conv1(x) -- x = self.avgpool(x)
        ####### 新加层的forward
        # x图片
        x = x.view(-1, 512)
        x = x.view(-1, sequence_length, 512)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        # d动态数据
        #d = d.view(-1, sequence_length, 16)
        #self.lstm2.flatten_parameters()
        #k, _ = self.lstm2(d)
        y = y.contiguous().view(-1, 128)
        #k = k.contiguous().view(-1, 128)
        # c = torch.cat([y,k],dim=0)
        #c = torch.add(y, k)
        # y = self.fc(y)
        y = self.fc(y)  # #因为接下来的self.fclass层的输入通道是
        return y


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length * srate), sequence_length * srate):
            idx.append(j)
        count += list_each_length[i]
    return idx

def frame_count(video, video_path):
    frames = []
    for v in video_path:
        mp4 = cv2.VideoCapture(video + v)  # 读取视频
        frame_count = mp4.get(7)
        frames.append(int(frame_count))
        #frames.append(int(np.ceil(frame_count/25)))
    return frames


def get_data():
    '''
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    '''

    video_path = os.listdir(video)
    anno = open(annotation, 'r')
    image_path = os.listdir(image)
    labels_path = anno.readlines()
    frames = frame_count(video, video_path)
    video_num = len(video_path)
    train_num_each = frames[0: math.floor(video_num * train_ratio)]
    val_num_each = frames[math.floor(video_num * train_ratio): math.floor(video_num * (train_ratio + val_ratio))]
    test_num_each = frames[math.floor(video_num * (train_ratio + val_ratio)):]
    train_num = sum(train_num_each)
    val_num = sum(val_num_each)
    train_paths = image_path[0:train_num]
    val_paths = image_path[train_num:train_num + val_num]
    test_paths = image_path[train_num + val_num:]
    labels_path = [l.split('\t')[0] for l in labels_path]  #####修改标号数字
    '''
    classes = labels_path
    classes = sorted(set(classes), key = classes.index)
    print(classes)
    '''
    #label_dict = {'Preparation\n': 0, 'CalotTriangleDissection\n': 1, 'ClippingCutting\n': 2,
                  #'GallbladderDissection\n': 3, 'GallbladderPackaging\n': 4, 'CleaningCoagulation\n': 5,
                  #'GallbladderRetraction\n': 6}
    label_dict = {'Idle':0, 'Suturing':1, 'Knot tying':2}
    #label_dict = {'Idle':0, 'Needle holding':1, 'Suture making':2, 'Suture handling':3, '1 knot':4, '2 knot':5, '3 knot':6, 'Idle Step':0}
    labels = [label_dict[l] for l in labels_path]

    train_labels = labels[0:train_num]
    val_labels = labels[train_num:train_num + val_num]
    test_labels = labels[train_num + val_num:]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    '''data_path = os.listdir(args.data)
    kdata = []
    for d in data_path:
        data = open(args.data + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        kdata = kdata + data

    kdata = np.asarray(kdata, dtype=np.float32)
    kdata = normalize(kdata, axis=0, norm='max')

    train_kdata = kdata[0:train_num, :]
    val_kdata = kdata[train_num:train_num + val_num, :]
    test_kdata = kdata[train_num + val_num:, :]'''

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    train_dataset = CholecDataset(image, train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(image, val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(image, test_paths, test_labels, test_transforms)

    '''train_dataset = CholecDataset(image, train_paths, train_labels, train_kdata, train_transforms)
    val_dataset = CholecDataset(image, val_paths, val_labels, val_kdata, test_transforms)
    test_dataset = CholecDataset(image, test_paths, test_labels, test_kdata, test_transforms)'''

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


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


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # num_train_we_use = 8000
    # num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]  # 训练数据开始位置
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    np.random.seed(0)
    np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j * srate)  # 训练数据位置，每一张图是一个数据

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j * srate)

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
    model = resnet_lstm()
    if use_gpu:
        model = model.cuda()

    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    '''
    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    record_np = np.zeros([epochs, 4])

    for epoch in range(epochs):
        np.random.seed(epoch)
        np.random.shuffle(train_we_use_start_idx)
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
            #inputs, labels_phase, kdata = data
            inputs, labels_phase = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_phase.cuda())
                #kdatas = Variable(kdata.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_phase)
                #kdatas = Variable(kdata)
            optimizer.zero_grad()
            #outputs = model.forward(inputs, kdatas)
            outputs = model.forward(inputs)
            outputs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)
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
                torch.save(copy.deepcopy(model.state_dict()), 'test.pth')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = train_corrects.cpu().numpy() / train_num
        train_average_loss = train_loss / train_num

        # begin eval
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        val_start_time = time.time()
        for data in val_loader:
            #inputs, labels_phase, kdata = data
            inputs, labels_phase = data
            #labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
            #kdata = kdata[(sequence_length - 1)::sequence_length]
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_phase.cuda())
                #kdatas = Variable(kdata.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_phase)
                #kdatas = Variable(kdata)

            if crop_type == 0 or crop_type == 1:
                #outputs = model.forward(inputs, kdatas)
                outputs = model.forward(inputs)
            elif crop_type == 5:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                #outputs = model.forward(inputs, kdatas)
                outputs = model.forward(inputs)
                outputs = outputs.view(5, -1, 3)
                outputs = torch.mean(outputs, 0)
            elif crop_type == 10:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                #outputs = model.forward(inputs, kdatas)
                outputs = model.forward(inputs)
                outputs = outputs.view(10, -1, 3)
                outputs = torch.mean(outputs, 0)

            #outputs = outputs[sequence_length - 1::sequence_length]

            _, preds = torch.max(outputs.data, 1)
            print(num)
            print(preds)
            print(labels)
            loss = criterion(outputs, labels)
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
    model_name = "lstm" \
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

    record_name = "lstm" \
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
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data()
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
