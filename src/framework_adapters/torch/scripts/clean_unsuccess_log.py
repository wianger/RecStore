import shutil
import glob
import os
import tqdm
all_dir = "/home/xieminhui/RecStore/src/framework_adapters/torch/benchmark/log"

all_exp_dirs = glob.glob(f"{all_dir}/*")

for exp_path in tqdm.tqdm(all_exp_dirs):
    exp_dir = exp_path

    log_files = glob.glob(f"{exp_dir}/*")

    for run_path in tqdm.tqdm(log_files):
        run_id = run_path.split("/")[-1]
        logfile = glob.glob(f'{run_path}/log')
        if len(logfile) != 1:
            print(run_path)
            continue
        assert len(logfile) == 1
        logfile = logfile[0]

        with open(logfile, "r") as f:
            lines = f.readlines()
        content = ''.join(lines)
        if content.find("Successfully xmh") != -1:
            # print(run_id, "find")
            pass
        else:
            print(run_id, run_path, "not find")
            shutil.rmtree(run_path)
