import shutil
import glob
import os
import tqdm
# exp_dir = "/home/xieminhui/RecStore/src/framework_adapters/torch/benchmark/log/exp3-KG-scale-3090"

exp_dir = "/home/xieminhui/RecStore/src/framework_adapters/torch/benchmark/log/0116-exp2-motiv-emb-3090"
log_files = glob.glob(f"{exp_dir}/*")

for run_path in tqdm.tqdm(log_files):
    run_id = run_path.split("/")[-1]
    logfile = glob.glob(f'{run_path}/log')
    assert len(logfile) == 1
    logfile = logfile[0]

    with open(logfile, "r") as f:
        lines = f.readlines()
    content = ''.join(lines)
    # print(content)
    if content.find("Successfully xmh") != -1:
        print(run_id, "find")
    else:
        print(run_id, "not find")
        shutil.rmtree(run_path)