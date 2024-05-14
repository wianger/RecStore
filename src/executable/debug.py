import torch as th

import sys
sys.path.append("/home/xieminhui/RecStore/src/python")  # nopep8
from recstore.utils import XLOG, Timer, GPUTimer, xmh_nvtx_range


a = th.load("/tmp/xmh.pkl")

emb = a[0]
grad = a[1]


t1= Timer("aaa")



Timer.StartReportThread()

while True:
	with th.no_grad():
		t1.start()
		emb.add_(grad)
		t1.stop()