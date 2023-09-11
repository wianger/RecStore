import enum
from ftplib import all_errors
from numpy import outer
import time
import argparse
import itertools
import os
import subprocess
import datetime
from bench_util import RemoteExecute,ParallelSSH,Pnuke
import exp_config
from exp_config import ALL_SERVERS_INCLUDING_NOT_USED, LOG_PREFIX, PROJECT_PATH
from zmq import SERVER
import concurrent.futures





def mount_master(hosts):
    pass


def config_each_server(hosts):
    ParallelSSH(
        hosts, f"sudo swapoff -a")


if __name__ == "__main__":
    Pnuke(ALL_SERVERS_INCLUDING_NOT_USED, "petps_server")
    Pnuke(ALL_SERVERS_INCLUDING_NOT_USED, "benchmark_client")

    exp_lists = []

    each = exp_config.ExpOverallSingle()
    each.SetLogDir(f'{LOG_PREFIX}/exp9-single-dram-rerun-1024')
    exp_lists.append(each)

    for i, each in enumerate(exp_lists):
        # mount NFS
        mount_master(
            [each for each in ALL_SERVERS_INCLUDING_NOT_USED if each != '10.0.2.130'])
        config_each_server(
            [each for each in ALL_SERVERS_INCLUDING_NOT_USED if each != '10.0.2.130'])

        print("=================-====================")
        print(f"Experiment {i}/{len(exp_lists)}: ", each.name)
        each.RunExperiment()

