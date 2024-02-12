import enum
from ftplib import all_errors
from numpy import outer
import time
import argparse
import itertools
import os
import subprocess
import datetime

from bench_util import RemoteExecute, ParallelSSH, Pnuke, GetHostName
import exp_config
from exp_config import ALL_SERVERS_INCLUDING_NOT_USED, LOG_PREFIX, PROJECT_PATH

from variables import *


def mount_master(hosts):
    pass


def config_each_server(hosts):
    ParallelSSH(
        hosts, f"sudo swapoff -a")


if GetHostName() == "node182":
    suffix = "A30"
else:
    suffix = "3090"

if __name__ == "__main__":
    exp_lists = []

    if suffix == 'A30':
        each = exp_config.ExpRecPerf()
        each.SetLogDir(f'{LOG_PREFIX}/fastpq0131-Rec-{suffix}')
        exp_lists.append(each)

        each = exp_config.ExpKGPerfA30()
        each.SetLogDir(f'{LOG_PREFIX}/fastpq0131-KG-perfA30-{suffix}')
        exp_lists.append(each)

        each = exp_config.ExpKGScalability()
        each.SetLogDir(f'{LOG_PREFIX}/fastpq0131-KG-scale-{suffix}')
        exp_lists.append(each)

        each = exp_config.ExpMotivationPerfEmb()
        each.SetLogDir(f'{LOG_PREFIX}/fastpq0131-exp2-motiv-emb-{suffix}')
        exp_lists.append(each)

    else:
        # # 用这个
        # each = exp_config.ExpMotivationDebug()
        # each.SetLogDir(f'{LOG_PREFIX}/0208-debugmicro-{suffix}')
        # exp_lists.append(each)

        
        each = exp_config.ExpRealMotivationPerfEmb()
        each.SetLogDir(f'{LOG_PREFIX}/0212-real-motiv-{suffix}')
        exp_lists.append(each)

        each = exp_config.ExpMotivationPerfEmb()
        # each.SetLogDir(f'{LOG_PREFIX}/0131-motiv-{suffix}')
        each.SetLogDir(f'{LOG_PREFIX}/0211-motiv-{suffix}')
        exp_lists.append(each)

        each = exp_config.ExpRecPerf()
        # each.SetLogDir(f'{LOG_PREFIX}/0128-Rec-{suffix}')  #实质上是0131重跑的
        each.SetLogDir(f'{LOG_PREFIX}/0210-Rec-{suffix}')
        exp_lists.append(each)

        each = exp_config.ExpKGScalability()
        each.SetLogDir(f'{LOG_PREFIX}/0210-KG-scale-{suffix}')
        each.SetFilter(lambda config: config['dataset'] == 'FB15k')
        exp_lists.append(each)


        each = exp_config.ExpKGScalability()
        
        each.SetLogDir(f'{LOG_PREFIX}/0131-KG-scale-{suffix}')
        exp_lists.append(each)


        each = exp_config.ExpKGSensitive()
        each.SetLogDir(f'{LOG_PREFIX}/0204-sen-{suffix}')
        exp_lists.append(each)

        # each = exp_config.ExpKGPerfDebug()
        # each.SetLogDir(f'{LOG_PREFIX}/0128-KG-debugomp{suffix}')
        # exp_lists.append(each)

        # # 别用下面的了
        # # each = exp_config.ExpMacroPerfEmb()
        # # each.SetLogDir(f'{LOG_PREFIX}/0117-exp1-macro-perf-emb-{suffix}')
        # # exp_lists.append(each)

    for i, each in enumerate(exp_lists):
        # mount NFS
        mount_master(
            [each for each in ALL_SERVERS_INCLUDING_NOT_USED if each != '127.0.0.1'])
        config_each_server(
            [each for each in ALL_SERVERS_INCLUDING_NOT_USED if each != '127.0.0.1'])

        print("=================-====================")
        print(f"Experiment {i}/{len(exp_lists)}: ", each.name,)

        each.RunExperiment()
