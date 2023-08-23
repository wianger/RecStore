from __future__ import absolute_import, division, print_function, unicode_literals

import queue as Q # import Python's Queue class for exception handling only
from multiprocessing import Queue, Process
#from utils.packets   import ServiceRequest
#from utils.utils  import debugPrint
import time
import numpy as np
import sys
import math
import random
import torch
#from scheduler import Scheduler


def model_arrival_times(args):
  print("lam = {}, size = {}".format(args.avg_arrival_rate, args.nepochs * args.num_batches))
  arrival_time_delays = np.random.poisson(lam  = args.avg_arrival_rate,
                                          size = args.nepochs * args.num_batches)
  return arrival_time_delays

def partition_requests(args, batch_size):
  batch_sizes = []

  while batch_size > 0:
    mini_batch_size = min(args.sub_task_batch_size, batch_size)
    batch_sizes.append(mini_batch_size)
    batch_size -= mini_batch_size

  return batch_sizes


def loadGenSleep( sleeptime ):
  if sleeptime > 0.0055:
    time.sleep(sleeptime)
  else:
    startTime = time.time()
    while (time.time() - startTime) < sleeptime:
      continue
  return

def send_request(client, dataset,
                 batch_id, epoch, batch_size, embedding_size):
  # print(f"[{time.time()}] batch_id = {batch_id}, epoch = {epoch}, batch_size = {batch_size}")
  indices = dataset.get(batch_size).to(torch.int64)
  GET_RATE = 2
  
  if random.random() < GET_RATE:
    _result = client.GetParameter(indices)
  else:
    client.PutParameter(indices, torch.rand(indices.shape[0], embedding_size))

def loadGenerator(args,
                  client,
                  dataset,
                  ):

  arrival_rate = args.avg_arrival_rate
  embedding_size = args.embedding_size
  batch_size = args.sub_task_batch_size
  epoch = 0
  exp_epochs = 0

  while exp_epochs < args.nepochs:
    for batch_id in range(args.num_batches):
      send_request(client = client,
                    dataset = dataset,
                    batch_id = batch_id,
                    epoch = epoch,
                    batch_size = batch_size,
                    embedding_size=embedding_size
                    )

      arrival_time = np.random.poisson(lam = arrival_rate, size = 1)
      loadGenSleep( arrival_time[0] / 1000. )
    epoch += 1
    exp_epochs += 1

  return
