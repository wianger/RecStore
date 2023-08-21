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


def model_batch_size_distribution(args):
  if args.batch_size_distribution == "normal":
    batch_size_distributions = np.random.normal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "lognormal":
    batch_size_distributions = np.random.lognormal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "fixed":
    batch_size_distributions = np.array([args.avg_mini_batch_size for _ in range(args.num_batches) ])

  elif args.batch_size_distribution == "file":
    percentiles = []
    batch_size_distributions = []
    with open(args.batch_dist_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        percentiles.append(float(line.rstrip()))

      for _ in range(args.num_batches):
        batch_size_distributions.append( int(percentiles[ int(np.random.uniform(0, len(percentiles))) ]) )

  for i in range(args.num_batches):
    batch_size_distributions[i] = int(max(min(batch_size_distributions[i], args.max_mini_batch_size), 1))
  return batch_size_distributions


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

def send_request(args, client, dataset,
                 batch_id, epoch, batch_size, sub_id, tot_sub_batches, embedding_size):
  # print(f"[{time.time()}] batch_id = {batch_id}, epoch = {epoch}, batch_size = {batch_size}, sub_id = {sub_id}, tot_sub_batches = {tot_sub_batches}")
  indices = dataset.get(batch_size)
  GET_RATE = 0.96
  
  if random.random() < GET_RATE:
    _result = client.GetParameter(indices)
  else:
    client.PutParameter(indices, torch.empty(indices.shape[0], embedding_size))

def loadGenerator(args,
                  client,
                  dataset,
                  #requestQueue,
                  #loadGeneratorReturnQueue,
                  #inferenceEngineReadyQueue,
                  #pidQueue,
                  #accelRequestQueue
                  ):
  # arrival_time_delays = model_arrival_times(args)
  batch_size_distributions = model_batch_size_distribution(args)
  # print(batch_size_distributions)
  cpu_sub_requests = 0
  cpu_requests = 0

  arrival_rate = args.avg_arrival_rate
  embedding_size = args.embedding_size

  epoch = 0
  exp_epochs = 0

  while exp_epochs < args.nepochs:
    for batch_id in range(args.num_batches):
      request_size = int(batch_size_distributions[batch_id])
      batch_sizes = partition_requests(args, request_size)
      for i, sub_batch_size in enumerate(batch_sizes):
        send_request(client = client,
                     dataset = dataset,
                     args = args,
                     batch_id = batch_id,
                     epoch = epoch,
                     batch_size = sub_batch_size,
                     sub_id = i,
                     tot_sub_batches = len(batch_sizes),
                     embedding_size=embedding_size
                     )
        cpu_sub_requests += 1
        cpu_requests += 1

      arrival_time = np.random.poisson(lam = arrival_rate, size = 1)
      loadGenSleep( arrival_time[0] / 1000. )
    epoch += 1
    exp_epochs += 1

  return
