from pprint import pprint
import subprocess
import os
import datetime
import time
import concurrent.futures

from bench_util import *
from bench_base import *
from variables import *


def ConvertHostNumaList2Host(host_numa_lists):
    return list(set([each[0] for each in host_numa_lists]))


class PerfEmbRun(LocalOnlyRun):
    def __init__(self, exp_id, run_id, log_dir, config, execute_host) -> None:
        self.execute_host = execute_host
        super().__init__(exp_id, run_id,
                         log_dir, config,  "python3 perf_emb.py",  "/home/xieminhui/RecStore/src/framework_adapters/torch", execute_host)

    def check_config(self,):
        super().check_config()

    def run(self):
        super().run()
        sleep_seconds = 0
        while True:
            ret = subprocess.run(
                f"grep 'Successfully xmh' {self.log_dir}/log >/dev/null 2>&1", shell=True).returncode
            if ret == 0:
                break
            time.sleep(5)
            sleep_seconds += 5

            if sleep_seconds > 30*60:
                for _ in range(100):
                    print("DEADLOCK in wait client finish")
                break

        print("tail down")
        Pnuke([self.execute_host], "perf_emb.py")


class ExpMacroPerfEmb(LocalOnlyExperiment):
    def __init__(self, ) -> None:
        NAME = "PerfEmbRun"
        COMMON_CONFIGS = {
            "num_workers": [1, 2, 4, 6, 8] if GetHostName() != "node182" else [0, 1, 2],
            "num_embs": [int(100*1e6), int(10*1e6)],
            "batch_size": [512, 1024, 2048, 4096,],
            "run_steps": [1000],
            "log_interval": [100],

            # "num_workers": [4],
            # "num_embs": [int(1*1e6)],
            # "run_steps": [100],
            # "log_interval": [10],

            "emb_choice": ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"],
        }

        self.name = NAME
        super().__init__(0, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortRuns(self, runs):
        return list(sorted(runs, key=lambda run: (run.config['num_embs'], run.config['batch_size'])))

    def _RunHook(self, previous_run, next_run):
        return

        
    def _PostprocessConfig(self, each_config, ):
        # don't use self
        pass
        # client_config['key_space_m'] *= WARM_UP_RATIO
        # client_config['key_space_m'] = int(client_config['key_space_m'])

    def _CreateRun(self, run_id, run_log_dir, run_config, execute_host):
        return PerfEmbRun(self.exp_id, run_id, run_log_dir,
                      run_config, execute_host)

    def _BeforeStartAllRun(self):
        print("pnuke perf_emb.py")
        Pnuke(ALL_SERVERS_INCLUDING_NOT_USED, "perf_emb.py")




###########################
###########################
###########################
###########################
class GNNRun(LocalOnlyRun):
    def __init__(self, exp_id, run_id, log_dir, config, execute_host) -> None:
        self.execute_host = execute_host
        super().__init__(exp_id, run_id,
                         log_dir, config,  "python3 dgl-ke-main.py", "/home/xieminhui/RecStore/src/framework_adapters/torch/python", execute_host)

    def check_config(self,):
        super().check_config()

    def run(self):
        super().run()
        sleep_seconds = 0
        while True:
            ret = subprocess.run(
                f"grep 'Successfully xmh' {self.log_dir}/log >/dev/null 2>&1", shell=True).returncode
            if ret == 0:
                break
            time.sleep(5)
            sleep_seconds += 5

            if sleep_seconds > 30*60:
                for _ in range(100):
                    print("DEADLOCK in wait client finish")
                break

        print("tail down")
        Pnuke([self.execute_host], "dgl-ke-main.py")


class GNNExperiment(LocalOnlyExperiment):
    def __init__(self, exp_id, common_config, execute_host) -> None:
        super().__init__(exp_id, common_config, execute_host)

    def _PostprocessConfig(self, each_config, ):
        # don't use self
        pass
        # client_config['key_space_m'] *= WARM_UP_RATIO
        # client_config['key_space_m'] = int(client_config['key_space_m'])

    def _CreateRun(self, run_id, run_log_dir, run_config, execute_host):
        return GNNRun(self.exp_id, run_id, run_log_dir,
                      run_config, execute_host)

    def _BeforeStartAllRun(self):
        print("pnuke dgl-ke-main.py")
        Pnuke(ALL_SERVERS_INCLUDING_NOT_USED, "dgl-ke-main.py")


COMMON_CLIENT_CONFIGS = {
    "no_save_emb": ['true'],
    "batch_size": [1000],
    "log_interval": [1000],
    "neg_sample_size": [200],
    "regularization_coef": [1e-07],
    "hidden_dim": [400],
    "gamma": [16.0],
    "lr": [0.01],
    "batch_size_eval": [16],
    "test": ["true"],
    "mix_cpu_gpu": ["true"],
}


class ExpOverallSingle(GNNExperiment):
    def __init__(self, ) -> None:
        NAME = "overall-single-machine"
        COMMON_CONFIGS = {
            "model_name": ["TransE_l1"],
            "dataset": ["FB15k", "Freebase"],
            # FB15k, FB15k-237, wn18, wn18rr and Freebase
            # "nr_gpus": [0, 1, 2, 3, 4, 5, 6, 7, 8] if GetHostName() != "node182" else [0, 1, 2, 3, 4],
            "nr_gpus": [1, 2, 3, 4, 5, 6, 7, 8] if GetHostName() != "node182" else [1, 2, 3, 4],

            "max_step": [10000],
            **COMMON_CLIENT_CONFIGS,
        }

        self.name = NAME
        super().__init__(1, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortRuns(self, runs):
        return list(sorted(runs, key=lambda run: run.config['dataset']))

    def _RunHook(self, previous_run, next_run):
        return