from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc_dropout import TrackerSiamFC


if __name__ == '__main__':
    net_path1 = 'pretrained/siamfc_alexnet_RGB_Dropout_e50.pth'
    net_path2 = 'pretrained/siamfc_alexnet_thermal_Dropout_e50.pth'
    tracker = TrackerSiamFC(net_path1=net_path1, net_path2=net_path2)

    root_dir = os.path.expanduser('~/RGB-t-Val')
    e = ExperimentOTB(root_dir, version='test_dropout')
    e.run(tracker)
    e.report([tracker.name])
