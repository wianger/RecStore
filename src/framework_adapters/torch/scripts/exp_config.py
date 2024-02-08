from pymemcache.client.base import Client
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


def GswLock():
    lock_file = "/tmp/xmh_gsw_lock"
    print(f"Want to lock the lock")

    while os.path.exists(lock_file):
        time.sleep(1)
    with open(lock_file, 'w') as file:
        file.write(
            'Hello, this is a new file!\nYou can add more content here.')
    print(f"Escape the lock")


def GswUnlock():
    print("unlock xmh_gsw_lock")
    lock_file = "/tmp/xmh_gsw_lock"
    try:
        os.remove(lock_file)
    except:
        pass


def MC_cas(key, old_value, new_value):
    mc = Client('localhost:11211')
    mc_old_value, cas_unique = mc.gets(key)
    if mc_old_value is None:
        ret = mc.set(key, new_value)
        assert ret
        return True
    mc_old_value = mc_old_value.decode('utf-8')
    if mc_old_value == old_value:
        ret = mc.cas(key, new_value, cas_unique)
        if ret == True:
            return True
    else:
        print(f"old value({old_value}) != readed_value({mc_old_value})",)
    return False


def MC_lock():
    print(f"Want to lock the lock")
    while True:
        ret = MC_cas("GPU_lock", "unlocked", "locked")
        if ret == True:
            break
        time.sleep(5)
    print(f"Escape the lock")


def MC_Unlock():
    print(f"Want to unlock the lock")
    while True:
        ret = MC_cas("GPU_lock", "locked", "unlocked")
        if ret == True:
            break
    print(f"unlock successfully")


def GPULock():
    if GetHostName() == "node182":
        GswLock()
    elif GetHostName() == "3090-server":
        MC_lock()
    else:
        assert 0


def GPUUnlock():
    if GetHostName() == "node182":
        GswUnlock()
    elif GetHostName() == "3090-server":
        MC_Unlock()
    else:
        assert 0


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
            "batch_size": [512, 1024, 2048, 3072, 4096,],
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
                        "TorchNativeStdEmb",
                        "KGExternelEmbedding",
                        "KnownShardedCachedEmbedding",
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

        if previous_run is not None:
            GPUUnlock()
        time.sleep(5)
        if next_run is not None:
            GPULock()
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
            # "num_workers": [4, 8] if GetHostName() != "node182" else [4],
            "num_workers": [8] if GetHostName() != "node182" else [4],

            "num_embs": [int(100*1e6),],
            # "batch_size": [512, 1024, 2048, 4096, 6144, 8192,],
            "batch_size": [128, 512, 1024, 1536, 2048,],
            "run_steps": [200],
            "log_interval": [100],

            "cache_ratio": [
                0.01,
                0.05,
                0.1,
            ],
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
                        "CppAsync",
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
        if previous_run is not None:
            GPUUnlock()
        # time.sleep(5)
        if next_run is not None:
            GPULock()
        return

    def _PostprocessConfig(self, each_config, ):
        assert each_config['cache_ratio'] * each_config['num_workers'] <= 1
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


class ExpMotivationDebug(LocalOnlyExperiment):
    def __init__(self, ) -> None:
        NAME = "debug micro benchmark"
        COMMON_CONFIGS = {
            "num_workers": [8] if GetHostName() != "node182" else [4],
            "num_embs": [int(100*1e6),],
            # "batch_size": [128, 512, 1024, 1536, 2048,],
            "batch_size": [512, 1024, 1536],
            "run_steps": [200],
            "log_interval": [100],
            "cache_ratio": [
                0.05,
            ],
            'binding2': [
                # {
                #     "distribution": ['uniform'],
                # },
                {
                    "distribution": ['zipf'],
                    "zipf_alpha": [
                        0.9,
                        # 0.99
                    ],
                },
            ],
            "binding": [
                {
                    "emb_choice": [
                        # "KGExternelEmbedding",
                        "KnownShardedCachedEmbedding",
                        # "TorchNativeStdEmb",
                    ]
                },
                # {

                #     "emb_choice": ["KnownLocalCachedEmbedding"],
                #     "backwardMode": [
                #         "PySync",
                #         # "CppSync",
                #         "CppAsyncV2",
                #         "CppAsync",
                #     ],
                # },
            ],
        }

        self.name = NAME
        super().__init__(33333, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        return list(sorted(configs, key=lambda config: (config['num_embs'], config['batch_size'])))

    def _RunHook(self, previous_run, next_run):
        # LocalNuke("perf_emb.py")
        LocalNukeAllPython()
        if previous_run is not None:
            GPUUnlock()
        # time.sleep(5)
        if next_run is not None:
            GPULock()
        return

    def _PostprocessConfig(self, each_config, ):
        assert each_config['cache_ratio'] * each_config['num_workers'] <= 1
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


class RecRun(LocalOnlyRun):
    def __init__(self, exp_id, run_id, log_dir, config, execute_host) -> None:
        self.execute_host = execute_host
        super().__init__(exp_id, run_id,
                         log_dir, config,  "python3 perf_rec_model.py", DIR_PATH, execute_host)

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
        LocalNuke("perf_rec_model.py")


class RecExperiment(LocalOnlyExperiment):
    def __init__(self, exp_id, common_config, execute_host) -> None:
        super().__init__(exp_id, common_config, execute_host)

    def _PostprocessConfig(self, each_config, ):
        # don't use self
        pass
        # client_config['key_space_m'] *= WARM_UP_RATIO
        # client_config['key_space_m'] = int(client_config['key_space_m'])

    def _CreateRun(self, run_id, run_log_dir, run_config, execute_host):
        return RecRun(self.exp_id, run_id, run_log_dir,
                      run_config, execute_host)

    def _BeforeStartAllRun(self):
        print("lnuke perf_rec_model.py")
        LocalNuke("perf_rec_model.py")


COMMON_CLIENT_CONFIGS = {
    "no_save_emb": ['true'],
    "neg_sample_size": [200],
    # "regularization_coef": [1e-07],
    "regularization_coef": [0],
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


class ExpKGScalability(GNNExperiment):
    def __init__(self, ) -> None:
        NAME = "overall-kg-scalability"
        COMMON_CONFIGS = {
            "model_name": [
                'TransE',
                'SimplE'

                # 'TransR',  OOM
                # 'RESCAL',  too slow
                # 'RotatE',  BUG

                # 'DistMult',
                # 'ComplEx',
                # 'SimplE'
            ],
            "binding": [
                {
                    "dataset": ["FB15k",],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05, 0.1],
                    # "batch_size": [1200],
                    "batch_size": [600, 1200, 1800, 2400, 3000] if GetHostName() == 'node182' else [1200],
                    "nr_gpus": [4, 8] if GetHostName() != "node182" else [4],
                },
                {
                    "dataset": ["Freebase"],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05, 0.1],
                    # "batch_size": [2000],
                    "batch_size": [600, 1200, 1800, 2400, 3000] if GetHostName() == 'node182' else [2000],
                    "nr_gpus": [4, 8] if GetHostName() != "node182" else [4],
                },
                # for scalability
                {
                    "dataset": ["FB15k", "Freebase"],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05,],
                    "batch_size": [2000],
                    "nr_gpus": [2, 3, 4, 5, 6, 7, 8] if GetHostName() != "node182" else [2, 3, 4],
                }
            ],
            "binding2": [
                # for debug performance
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": [
                        "CppAsyncV2",
                        "PySync",
                        "CppAsync"],
                },
                {
                    "use_my_emb": ["false"],
                    "cached_emb_type": ['None'],
                    "backwardMode": ["CppSync"],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KGExternelEmbedding',
                                        # "TorchNativeStdEmb",
                                        'KnownShardedCachedEmbedding'],
                    "backwardMode": ["PySync"],
                },
            ],
            "max_step": [500],
            "log_interval": [100],
            **COMMON_CLIENT_CONFIGS,
        }

        self.name = NAME
        super().__init__(10, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        need_run = []
        for each in configs:
            if GetHostName() == 'node182' and each['dataset'] == 'Freebase':
                print("pass Freebase")
                continue
            print(each)
            need_run.append(each)

        return list(sorted(need_run, key=lambda each: each['dataset']))

    def _RunHook(self, previous_run, next_run):
        LocalExecute('rm -rf /tmp/cached_tensor_*', '')
        print("lnuke dgl-ke-main.py")
        LocalNuke("dgl-ke-main.py")
        LocalNukeAllPython()
        if previous_run is not None:
            GPUUnlock()
        time.sleep(5)
        if next_run is not None:
            GPULock()
        return


class ExpKGSensitive(GNNExperiment):
    def __init__(self, ) -> None:
        NAME = "overall-kg-sensitive"
        COMMON_CONFIGS = {
            "model_name": [
                'TransE',
                # 'SimplE'
                # 'TransR',  OOM
                # 'RESCAL',  too slow
                # 'RotatE',  BUG

                # 'DistMult',
                # 'ComplEx',
                # 'SimplE'
            ],
            "binding": [
                {
                    "dataset": ["FB15k",],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05,],
                    "batch_size": [1200],
                    "nr_gpus": [8]
                },
                {
                    "dataset": ["Freebase"],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05,],
                    "batch_size": [2000],
                    "nr_gpus": [8]
                },
            ],
            "binding2": [
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": [
                        "PySync",
                    ],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": [
                        "CppAsyncV2",
                        "CppAsync"],
                    "L": [2, 4, 6, 8, 10, 14],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": [
                        "CppAsyncV2",
                        "CppAsync"],
                    "nr_background_threads": [2, 4, 6, 8, 12, 16, 24, 32],
                },
                {
                    "use_my_emb": ["false"],
                    "cached_emb_type": ['None'],
                    "backwardMode": ["CppSync"],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KGExternelEmbedding',
                                        # "TorchNativeStdEmb",
                                        'KnownShardedCachedEmbedding'],
                    "backwardMode": ["PySync"],
                },
            ],
            "max_step": [500],
            "log_interval": [100],
            **COMMON_CLIENT_CONFIGS,
        }

        self.name = NAME
        super().__init__(1230, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        need_run = []
        for each in configs:
            if GetHostName() == 'node182' and each['dataset'] == 'Freebase':
                print("pass Freebase")
                continue
            print(each)
            need_run.append(each)

        return list(sorted(need_run, key=lambda each: each['dataset']))

    def _RunHook(self, previous_run, next_run):
        LocalExecute('rm -rf /tmp/cached_tensor_*', '')
        print("lnuke dgl-ke-main.py")
        LocalNuke("dgl-ke-main.py")
        LocalNukeAllPython()
        if previous_run is not None:
            GPUUnlock()
        time.sleep(5)
        if next_run is not None:
            GPULock()
        return


class ExpKGPerfDebug(GNNExperiment):
    def __init__(self, ) -> None:
        NAME = "debug-kg"
        COMMON_CONFIGS = {
            "model_name": [
                'TransE',
            ],
            "binding": [
                {
                    "dataset": ["FB15k",],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05, ],
                    "batch_size": [1200],
                },
                {
                    "dataset": ["Freebase"],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05],
                    "batch_size": [2000],
                }
            ],
            "binding2": [
                # for debug performance
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": ["CppAsyncV2"],
                    "update_cache_use_omp": [0, 1],
                    "update_pq_use_omp": [0, 1],
                },
            ],
            "nr_gpus": [8] if GetHostName() != "node182" else [4],
            "max_step": [500],
            "log_interval": [100],
            **COMMON_CLIENT_CONFIGS,
        }

        self.name = NAME
        super().__init__(11, COMMON_CONFIGS,
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
        if previous_run is not None:
            GPUUnlock()
        time.sleep(5)
        if next_run is not None:
            GPULock()
        return


class ExpKGPerfA30(GNNExperiment):
    def __init__(self, ) -> None:
        NAME = "kg a30"
        COMMON_CONFIGS = {
            "model_name": [
                'TransE',
                'SimplE'
            ],
            "binding": [
                {
                    "dataset": ["FB15k",],
                    "hidden_dim": [400],
                    "cache_ratio": [0.05, 0.1],
                    "batch_size": [1200],
                    "nr_gpus": [4],
                },
            ],
            "binding2": [
                # for debug performance
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KnownLocalCachedEmbedding'],
                    "backwardMode": [
                        "CppAsyncV2",
                        "PySync",
                        "CppAsync"],
                },
                {
                    "use_my_emb": ["false"],
                    "cached_emb_type": ['None'],
                    "backwardMode": ["CppSync"],
                },
                {
                    "use_my_emb": ["true"],
                    "cached_emb_type": ['KGExternelEmbedding',
                                        "TorchNativeStdEmb",
                                        'KnownShardedCachedEmbedding'],
                    "backwardMode": ["PySync"],
                },
            ],
            "max_step": [500],
            "log_interval": [100],
            **COMMON_CLIENT_CONFIGS,
        }

        self.name = NAME
        super().__init__(13, COMMON_CONFIGS,
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
        if previous_run is not None:
            GPUUnlock()
        time.sleep(5)
        if next_run is not None:
            GPULock()
        return


class ExpRecPerf(RecExperiment):
    def __init__(self, ) -> None:
        NAME = "rec perf a30"
        COMMON_CONFIGS = {
            "with_nn": [
                '512,256,1',
            ],
            "binding": [
                {
                    "dataset": ["criteo",],
                    # "cache_ratio": [0.05,],
                    "cache_ratio": [0.05, 0.1] if GetHostName() == "node182" else [0.05],
                    "batch_size": [256, 512, 1024, 2048],
                    "num_workers": [4, 8] if GetHostName() != "node182" else [4],
                },
                {
                    "dataset": ["avazu"],
                    # "cache_ratio": [0.05,],
                    "cache_ratio": [0.05, 0.1] if GetHostName() == "node182" else [0.05],
                    "batch_size": [256, 512, 1024, 2048],
                    "num_workers": [4, 8] if GetHostName() != "node182" else [4],
                },

                # for scalability
                {
                    "dataset": ["avazu"],
                    "cache_ratio": [0.05],
                    "batch_size": [2048],
                    "num_workers": [2, 4, 6, 8] if GetHostName() != "node182" else [1, 2, 3, 4],
                }
            ],
            "binding2": [
                {
                    "emb_choice": [
                        "TorchNativeStdEmb",
                        "KGExternelEmbedding",
                        "KnownShardedCachedEmbedding",
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
            "run_steps": [300],
            "log_interval": [100],
        }

        self.name = NAME
        super().__init__(12, COMMON_CONFIGS,
                         "127.0.0.1")

    def _SortConfigs(self, configs):
        for each in configs:
            print(each)
        return list(sorted(configs, key=lambda each: each['dataset']))

    def _RunHook(self, previous_run, next_run):
        LocalExecute('rm -rf /tmp/cached_tensor_*', '')
        print("lnuke perf_rec_model.py")
        LocalNuke("perf_rec_model.py")
        LocalNukeAllPython()
        if previous_run is not None:
            GPUUnlock()
        time.sleep(5)
        if next_run is not None:
            GPULock()
        return
