from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e10.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('~/data/GOT-10k/train_i')
    e = ExperimentOTB(root_dir, version=2019)
    e.run(tracker)
    e.report([tracker.name])
