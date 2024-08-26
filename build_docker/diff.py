import os
import sys

gt = os.listdir("test_rendered/")
ours = os.listdir("videos_volume")

gt = set(gt)
ours = set(ours)

import pdb
pdb.set_trace()