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


DIR_PATH = "/home/xieminhui/RecStore/src/framework_adapters/torch/python"


class PerfEmbRun(LocalOnlyRun):
    def __init__(self, exp_id, run_id, log_dir, config, execute_host) -> None:
        self.execute_host = execute_host
        super().__init__(exp_id, run_id,
                         log_dir, config,  "python3 perf_emb.py",  DIR_PATH, execute_host)

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

            if sleep_seconds > 60*60:
                for _ in range(100):
                    print("DEADLOCK in wait client finish")
                break

        print("tail down")
        Pnuke([self.execute_host], "perf_emb.py")


class ExpMacroPerfEmb(LocalOnlyExperiment):
    def __init__(self, ) -> None:
        NAME = "PerfEmbRun"
        COMMON_CONFIGS = {
            "num_workers": [4, 8] if GetHostName() != "node182" else [4],

            "num_embs": [int(100*1e6),],
            "batch_size": [512, 1024, 2048, 4096,],
            "run_steps": [300],
            "log_interval": [100],

            # "batch_size": [2048,],
            # "num_embs": [int(100*1e6),],
            # "run_steps": [300],
            # "log_interval": [100],

            'binding2': [
                {
                    "distribution": ['uniform'],
                },
                {
                    "distribution": ['zipf'],
                    "zipf_alpha": [
                        0.9,
                        0.99
                    ],
                },
            ],

            "binding": [
                {
                    "emb_choice": [
                        # "TorchNativeStdEmb",
                        "KGExternelEmbedding",
                        "KnownShardedCachedEmbedding",
                    ]
                },
                {

                    "emb_choice": ["KnownLocalCachedEmbedding"],
                    "backwardMode": [
                        # "PySync",
                        # "CppSync",
                        "CppAsyncV2",
                        # "CppAsync",
                    ],
                    # "backgrad_init": [
                    #     "cpu", "both"
                    # ]
                },
            ],
        }

        self.name = NAME
        super().__init__(0, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        return list(sorted(configs, key=lambda config: (config['num_embs'], config['batch_size'])))

    def _RunHook(self, previous_run, next_run):
        # LocalNuke("perf_emb.py")
        LocalNukeAllPython()
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
        # LocalNuke("perf_emb.py")
        LocalNukeAllPython()


class ExpMotivationPerfEmb(LocalOnlyExperiment):
    def __init__(self, ) -> None:
        NAME = "MotivationPerfEmb"
        COMMON_CONFIGS = {
            "num_workers": [4,] if GetHostName() != "node182" else [4],
            "num_embs": [int(100*1e6),],
            "batch_size": [512, 1024, 2048, 4096,],
            "run_steps": [300],
            "log_interval": [100],

            "cache_ratio": [0.1, 0.2],
            'binding2': [
                {
                    "distribution": ['uniform'],
                },
                {
                    "distribution": ['zipf'],
                    "zipf_alpha": [
                        0.9,
                        0.99
                    ],
                },
            ],
            "binding": [
                {
                    "emb_choice": [
                        "KGExternelEmbedding",
                        "KnownShardedCachedEmbedding",
                        "TorchNativeStdEmb",
                    ]
                },
                {

                    "emb_choice": ["KnownLocalCachedEmbedding"],
                    "backwardMode": [
                        "PySync",
                        # "CppSync",
                        "CppAsyncV2",
                        # "CppAsync",
                    ],
                },
            ],
        }

        self.name = NAME
        super().__init__(3, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        return list(sorted(configs, key=lambda config: (config['num_embs'], config['batch_size'])))

    def _RunHook(self, previous_run, next_run):
        # LocalNuke("perf_emb.py")
        LocalNukeAllPython()
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
        # LocalNuke("perf_emb.py")
        LocalNukeAllPython()


###########################
###########################
###########################
###########################
class GNNRun(LocalOnlyRun):
    def __init__(self, exp_id, run_id, log_dir, config, execute_host) -> None:
        self.execute_host = execute_host
        super().__init__(exp_id, run_id,
                         log_dir, config,  "python3 dgl-ke-main.py", DIR_PATH, execute_host)

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

            if sleep_seconds > 60*60:
                for _ in range(100):
                    print("DEADLOCK in wait client finish")
                break

        print("tail down")
        LocalNuke("dgl-ke-main.py")


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
        print("lnuke dgl-ke-main.py")
        LocalNuke("dgl-ke-main.py")


COMMON_CLIENT_CONFIGS = {
    "no_save_emb": ['true'],
    "neg_sample_size": [200],
    "regularization_coef": [1e-07],
    "gamma": [16.0],
    "lr": [0.01],
    "batch_size_eval": [16],
    "test": ["false"],
    "mix_cpu_gpu": ["true"],
}

['Freebase', 'FB15k', 'FB15k-237', 'wn18',
    'wn18rr', 'wikikg2', 'biokg', 'wikikg90M']

['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
 'RESCAL', 'DistMult', 'ComplEx', 'RotatE',
 'SimplE'],


class ExpOverallSingle(GNNExperiment):
    def __init__(self, ) -> None:
        NAME = "overall-single-machine"
        COMMON_CONFIGS = {
            "model_name": ["TransE_l1"],
            "binding": [
                {
                    "dataset": ["FB15k",],
                    "hidden_dim": [400],
                },
                # {
                #     "dataset": ["Freebase"],
                #     "hidden_dim": [100],
                # }
            ],
            "binding2": [
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": ["PySync"],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": ["CppSync"],
                },
                {
                    "use_my_emb": ["false"],
                    "cached_emb_type": ['None'],
                    "backwardMode": ["CppSync"],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KGExternelEmbedding', 'KnownShardedCachedEmbedding'],
                    "backwardMode": ["PySync"],
                },
            ],


            # FB15k, FB15k-237, wn18, wn18rr and Freebase
            # "nr_gpus": [0, 1, 2, 3, 4, 5, 6, 7, 8] if GetHostName() != "node182" else [0, 1, 2, 3, 4],

            "nr_gpus": [4, 8] if GetHostName() != "node182" else [4],
            "batch_size": [600, 1200, 1800, 3000, 4800, 6600, 8400],
            "cache_ratio": [0.1, 0.2],

            # "nr_gpus": [2, 4],
            # "batch_size": [500, 1000, 2000,],
            # "cache_ratio": [0.05, 0.1, 0.4],

            "max_step": [500],
            "log_interval": [200],
            **COMMON_CLIENT_CONFIGS,
        }

        self.name = NAME
        super().__init__(1, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        for each in configs:
            print(each)
        return list(sorted(configs, key=lambda each: each['dataset']))

    def _RunHook(self, previous_run, next_run):
        LocalExecute('rm -rf /tmp/cached_tensor_*', '')
        print("lnuke dgl-ke-main.py")
        LocalNuke("dgl-ke-main.py")
        LocalNukeAllPython()
        return
