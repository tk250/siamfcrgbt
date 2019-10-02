from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = os.path.expanduser('~\data\GOT-10k')
    seqs_i = GOT10k(root_dir, subset='train_i', return_meta=False,
                  visible=False)
    seqs_v = GOT10k(root_dir, subset='train_i', return_meta=False,
                  visible=True)

    tracker = TrackerSiamFC()
    tracker.train_over(seqs_i, seqs_v)
