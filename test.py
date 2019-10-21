from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC
from siamfc_dense import TrackerSiamFCDense
from siamfc_DenseSSMA import TrackerSiamFCDenseSSMA


if __name__ == '__main__':
    net_path1 = 'pretrained/siamfc_alexnet_SSMA_new_e50.pth'

    tracker1 = TrackerSiamFC(net_path=net_path1, name='SSMA')

    net_path2 = 'pretrained/siamfc_alexnet_Dense_new_e50.pth'

    tracker2 = TrackerSiamFCDense(net_path=net_path2, name='DenseMMF')

    net_path3 = 'pretrained/siamfc_alexnet_Dense_SSMA_e50.pth'

    tracker3 = TrackerSiamFCDenseSSMA(net_path=net_path3, name='Dense&SSMA')

    root_dir = os.path.expanduser('~/RGB-t-Val')
    e = ExperimentOTB(root_dir, version='test_all')
    e.run(tracker1)
    e.run(tracker2)
    e.run(tracker3)

    e.report([tracker1.name, tracker2.name, tracker3.name])
