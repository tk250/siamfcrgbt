from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

import ops
from backbones import AlexNetV1, SiameseNetRGB, SiameseNetThermal
from heads import SiamFC
from losses import BalancedLoss
from datasets import Pair
from transforms import SiamFCTransforms
from got10k.utils.metrics import rect_iou


__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z1, x1, z2, x2):
        z = self.backbone(z1, z2)
        x = self.backbone(x1, x2)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path1=None, net_path2=None,**kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net1 = Net(
            backbone=SiameseNetRGB(),
            head=SiamFC(self.cfg.out_scale))
        self.net2 = Net(
            backbone=SiameseNetThermal(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net1)
        
        # load checkpoint if provided
        if net_path1 is not None:
            self.net1.load_state_dict(torch.load(
                net_path1, map_location=lambda storage, loc: storage))
        self.net1 = self.net1.to(self.device)

        if net_path2 is not None:
            self.net2.load_state_dict(torch.load(
                net_path2, map_location=lambda storage, loc: storage))
        self.net2 = self.net2.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer1 = optim.SGD(
            self.net1.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        self.optimizer2 = optim.SGD(
            self.net2.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer1, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, visible_img, infrared_img, box):
        # set to evaluation mode
        self.net1.eval()
        self.net2.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar visible image
        self.avg_color = np.mean(visible_img, axis=(0, 1))
        z1 = ops.crop_and_resize(
            visible_img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar infrared image
        self.avg_color = np.mean(infrared_img, axis=(0, 1))
        z2 = ops.crop_and_resize(
            infrared_img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z1 = torch.from_numpy(z1).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        z2 = torch.from_numpy(z2).unsqueeze(2).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net1.backbone(z1)
    
    @torch.no_grad()
    def update(self, visible_img, infrared_img, validation=False, f=0):
        # set to evaluation mode
        self.net1.eval()
        self.net2.eval()

        # make dropouts work if we want to calculate uncertainity
        if validation == True:
            self.net1.apply(dropout)
            self.net2.apply(dropout)

        # visible search images
        x1 = [ops.crop_and_resize(
            visible_img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color, f=f) for f in self.scale_factors]
        x1 = np.stack(x1, axis=0)
        x1 = torch.from_numpy(x1).to(
            self.device).permute(0, 3, 1, 2).float()

        # infrared search images
        x2= [ops.crop_and_resize(
            infrared_img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x2 = np.stack(x2, axis=0)
        x2 = torch.from_numpy(x2).unsqueeze(3).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        response_RGB = self.net1.backbone(x1)
        response_thermal = self.net2.backbone(x2)
        RGB_box = self._get_box(response_RGB)
        thermal_box = self._get_box(response_thermal)

        return [RGB_box, thermal_box]
    
    def track(self, visible_files, infrared_files, box, visualize=False):
        frame_num = len(visible_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        RGB_boxes = np.zeros((frame_num, 4))
        RGB_boxes[0] = box
        thermal_boxes = np.zeros((frame_num, 4))
        thermal_boxes[0] = box
        times = np.zeros(frame_num)

        for f, visible_file in enumerate(visible_files):
            visible_img = ops.read_image(visible_file)
            infrared_img = cv2.imread(infrared_files[f], cv2.IMREAD_GRAYSCALE)

            begin = time.time()
            if f == 0:
                self.init(visible_img, infrared_img, box)
            else:
                res_boxes = self.update(visible_img, infrared_img, f)
                RGB_intersection = 0
                thermal_intersection = 0
                for c in range(30):
                    dropout_boxes = self.update(visible_img, infrared_img, validation=True)
                    RGB_intersection += rect_iou(dropout_boxes[0], res_boxes[0])
                    thermal_intersection += rect_iou(dropout_boxes[1], res_boxes[1])
                if RGB_intersection < thermal_intersection:
                    boxes[f, :] = res_boxes[0]
                else:
                    boxes[f, :] = res_boxes[1]
                    
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(visible_img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z1 = batch[0].to(self.device, non_blocking=self.cuda)
        x1 = batch[1].to(self.device, non_blocking=self.cuda)
        z2 = batch[2].to(self.device, non_blocking=self.cuda)
        x2 = batch[3].to(self.device, non_blocking=self.cuda)


        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z1, x1, z2, x2)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs1, seqs2, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs_rgb=seqs1,
            seqs_i=seqs2,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_thermal_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

    def _get_box(self, x):
        responses = self.net1.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))


        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels


def dropout(module):
    if type(module) == nn.Dropout:
        print('Set module {} to train mode.'.format(m))
        m.train(True)
