import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
from full.stage2_k_2.stage2_k import res34_tcn
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder
from torch.nn import DataParallel
from torch.nn import functional as F
import os
import glob
import sys
from PIL import Image
import cv2
import imageio
import numpy as np
import argparse
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-s', '--seq', default=16, type=int, help='sequence length, default 4')
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

parser.add_argument('--annotation', default='F:/MISAW_test/Procedural decription/annotation.txt', type=str,
                    help='annotation path')
parser.add_argument('--video', default='F:/MISAW_test/Video/', type=str, help='Video path')
parser.add_argument('--image', default='F:/MISAW_test/Image/', type=str, help='Video to images save path')
parser.add_argument('--data', default='F:/MISAW_test/Kinematic/', type=str, help='Video to kdata save path')
parser.add_argument('--srate', default=5, type=int, help='sample rate')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
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

video_path = 'F:/MISAW_test/Video/'
kinematic_path = 'F:/MISAW_test/Kinematic/'
list_of_video_file = glob.glob(os.path.join(video_path, "*.mp4*"))
list_of_kinematic_file = glob.glob(os.path.join(kinematic_path, "*.txt*"))
output_path = 'F:/MISAW_test/Outputs'

assert len(list_of_video_file) > 0, "No video files were found in the input folder provided!!"
assert len(list_of_kinematic_file) > 0, "No kinematic files were found in the input folder provided!!"

output_File = open(output_path + "/outputFile.txt", "a")
for video_filename in list_of_video_file:
    output_File.write('--> Found video file named: {0}\n'.format(video_filename))

for kinematic_filename in list_of_kinematic_file:
    output_File.write('--> Found kinematic file named: {0}\n'.format(kinematic_filename))

output_File.close()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sequence_length = 16
test_batch_size = 128
srate = 5
model_name = 'F:/TCN/full/stage2_k_2/tcn_epoch_100_length_16_opt_1_mulopt_1_flip_0_crop_1_batch_128_train_9860_val_8025.pth'

model_pure_name, _ = os.path.splitext(model_name)

use_gpu = torch.cuda.is_available()

print('sequence length : {:6d}'.format(sequence_length))
print('test batch size : {:6d}'.format(test_batch_size))
print('name of this model: {:s}'.format(model_name))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.crop((0, 0, img.size[0] / 2, img.size[1]))
            return img.convert('RGB')


def frame_count(video, video_path):
    frames = []
    for v in video_path:
        mp4 = cv2.VideoCapture(video + v)  # 读取视频. 例如：F:/Feature/MISAW/Video/1_1.mp4
        frame_count = mp4.get(7)  # 视频文件中的帧数
        frames.append(int(frame_count))  # 每个视频的帧数组成的数组
        # frames.append(int(np.ceil(frame_count/25)))
    return frames


class MISAWDataset(Dataset):
    def __init__(self, image, file_paths, file_labels, kdata,
                 transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels
        self.image = image
        self.kdata = kdata
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.image + self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        data = self.kdata[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase, data

    def __len__(self):
        return len(self.file_paths)


def get_useful_end_idx(sequence_length, test_num):
    idx = []
    '''for j in range(0, (test_num + 1 - sequence_length * srate), sequence_length * srate):  # start ID在最后几个构不成一个batch
        idx.append(j)
    return idx'''
    for j in range(test_num-1, sequence_length*srate-2, -sequence_length*srate):
        for i in range(srate):
            idx.append(j-i)
    return idx


def get_data():
    video_path = os.listdir(video)
    test_path = video_path

    anno = open(annotation, 'r')
    image_path = os.listdir(image)
    labels_path = anno.readlines()
    frames = frame_count(video, video_path)
    val_num_each = frame_count(video, test_path)
    val_num = sum(val_num_each)

    val_paths = image_path[0:val_num]
    labels_path = [l.split('\t')[1] for l in labels_path]
    label_dict = {'Idle': 0, 'Needle holding': 1, 'Suture making': 2, 'Suture handling': 3, '1 knot': 4, '2 knot': 5,
                  '3 knot': 6, 'Idle Step': 0}
    labels = [label_dict[l] for l in labels_path]
    val_labels = labels[0: val_num]

    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))

    val_labels = np.asarray(val_labels, dtype=np.int64)

    data_path = os.listdir(args.data)
    kdata = []
    for d in data_path:
        data = open(args.data + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        kdata = kdata + data

    kdata = np.asarray(kdata, dtype=np.float32)
    kdata = normalize(kdata, axis=0, norm='max')

    val_kdatas = kdata[0:val_num]

    # torchvision.transforms是pytorch中的图像预处理包
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    val_datasets = []
    ind = 0
    for i in range(len(video_path)):
        val_path = val_paths[ind: ind + val_num_each[i]]
        val_label = val_labels[ind: ind + val_num_each[i]]
        val_kdata = val_kdatas[ind: ind + val_num_each[i]]
        ind += val_num_each[i]
        val_dataset = MISAWDataset(image, val_path, val_label, val_kdata, test_transforms)
        val_datasets.append(val_dataset)
    return val_datasets, val_num_each


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def test_model(test_dataset):
    num_test = len(test_dataset)
    test_useful_end_idx = get_useful_end_idx(sequence_length, num_test)
    test_idx = []
    for i in test_useful_end_idx:
        for j in range(sequence_length):
            test_idx.append(i - j * srate)
    test_idx.reverse()
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=SeqSampler(test_dataset, test_idx),
        # sampler=test_idx,
        num_workers=0,
        pin_memory=False
    )
    model = res34_tcn()
    model = DataParallel(model)
    model.load_state_dict(torch.load(model_name))
    # model = model.module
    # model = DataParallel(model)

    if use_gpu:
        model = model.cuda()
    # model = DataParallel(model)
    # model = model.module

    model.eval()

    all_preds_s = []

    num = 0
    with torch.no_grad():
        for data in test_loader:
            num = num + 1
            inputs, _, kdatas = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                kdatas = Variable(kdatas.cuda())
            else:
                inputs = Variable(inputs)
                kdatas = Variable(kdatas)

            outputs_s = model.forward(inputs, kdatas)

            #outputs_s = outputs_s[-1, (sequence_length - 1):: sequence_length]
            outputs_s = outputs_s[-1]
            outputs_s = F.softmax(outputs_s, dim=-1)

            _, preds_s = torch.max(outputs_s.data, -1)

            for j in range(preds_s.shape[0]):
                all_preds_s.append(preds_s[j].data.item())

    return all_preds_s


def main():
    step_dict = {0: 'Idle', 1: 'Needle holding', 2: 'Suture making', 3: 'Suture handling', 4: '1 knot', 5: '2 knot',
                 6: '3 knot'}

    test_datasets, num_each = get_data()
    for i in range(len(num_each)):
        all_preds_s = test_model(test_datasets[i])
        preds_s = np.asarray(all_preds_s).reshape((sequence_length, -1, srate))
        s = []
        for k in range(preds_s.shape[2]):
            for m in range(preds_s.shape[0]):
                for j in range(preds_s.shape[1]):
                    s.append(preds_s[m, j, k])
        fname = str(i)
        f = output_path + '/' + fname + '_Results_Multi.txt'  ################记得改掉split

        fnum = 0
        with open(f, 'w') as of:
            of.write(
                'Frame' + '\t' + 'Step' + '\n')
            fnum = fnum + 1
            for i in range(num_each[i]-len(s)):
                of.write(str(fnum) + '\t' + 'Idle' + '\n')
                fnum = fnum + 1
            for i in range(len(s)):
                ous = step_dict[s[i]]
                of.write(str(fnum) + '\t' + ous + "\n")
                fnum = fnum + 1

        print('finish writing ' + fname)


if __name__ == "__main__":
    main()

print('Done')
print()
