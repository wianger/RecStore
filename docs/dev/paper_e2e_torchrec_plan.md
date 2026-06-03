# RecStore vs TorchRec 投稿级端到端实验设计

本文档定义 RecStore 与 TorchRec 的投稿级端到端对比实验。已有存储层实验不在这里重复展开；这里重点补齐 PyTorch/model 层，以及与 RDMA 参数服务器路径的边界关系。

## 实验边界

所有结果必须按层标注：

- `PyTorch/model`：通过 `model_zoo/rs_demo` 跑训练 step，统计 `samples/s`、`step_total_ms`、`embed_lookup_local_ms`、`sparse_update_ms`。这是 RecStore 与 TorchRec 的主对比层。
- `PS/network`：通过 `tools/benchmarks/run_benchmark_ps.py` 跑参数服务器和 RDMA 路径，统计 `M keys/s`。该层只能解释传输和服务器路径，不直接写成端到端训练加速。
- `storage-only`：已有 KVEngine/YCSB/存储微基准，只用于解释后端行为，不外推成模型吞吐。

当前机器只有 1 张 GPU，因此现有报告中的 2proc 行全部是 `skipped`。投稿前必须在 2/4/8 GPU 节点补跑多卡扩展性；否则论文只能声称单 GPU 端到端结果和 PS/network RDMA 校准。

## 相关论文实验章节抽象

近期推荐系统训练/embedding 存储相关论文使用 TorchRec 或 DLRM/TorchRec 生态做 baseline 时，实验设计通常覆盖以下维度：

- 端到端训练吞吐和 step latency，而不是只给 embedding lookup 微基准。
- HBM、UVM/UVM cache、CPU/host memory、SSD/offload 等容量层级。
- embedding table 行数、embedding dimension、batch size、访问分布和缓存命中率敏感性。
- 单机多 GPU、跨节点、参数服务器或通信路径扩展性。
- embedding lookup、pooling、backward、sparse update、data loading 的时间分解。
- ablation：传输层、索引结构、prefetch/cache 深度、异步策略、调度策略。
- OOM、timeout、startup failure、skipped 的独立记录，不能删除失败行后只报告成功均值。

对 RecStore 来说，TorchRec baseline 至少应包括：

- `TorchRec-HBM`：GPU HBM 纯 TorchRec 路径，作为小模型/高 batch 场景的强 baseline。
- `TorchRec-UVMCache`：TorchRec UVM caching 路径，作为大 embedding table 场景的主 baseline。
- 可选 `TorchRec-UVM` 或 host memory 路径：如果当前 TorchRec 版本和 runner 可稳定暴露该模式，再作为容量消融；否则不要手写等价结论。

## 已有数据状态

最终合并报告目录：

```text
/nas/home/shq/docker/rs_demo/paper_e2e_full_report_0601
```

关键文件：

- `paper_e2e_report.tex`：当前 LaTeX 报告，未生成 PDF，因为环境缺少 LaTeX 编译器。
- `summary_e2e.csv`：499 行，其中 387 `ok`、96 `skipped`、16 `failed`。
- `summary_gap.csv`：40 个可与 TorchRec-HBM/UVMCache 配对的配置。
- `summary_ps_network.csv`：120 条 RDMA PS/network 行。

现有主结论只能这样表述：

- 在 40 个可配对配置中，最佳 RecStore 路径多数配置快于 TorchRec-HBM 和 TorchRec-UVMCache。
- 小 batch，特别是 `batch_size <= 1024`，RecStore 相对 TorchRec-UVMCache 优势更稳定。
- 大 batch，特别是 `batch_size >= 4096`，TorchRec-UVMCache 经常更强，必须单独报告，不能被小 batch 均值掩盖。
- `prefetch_depth > 0` 当前是负面消融项，不能作为主优化结果。
- RDMA 当前只有 PS/network 层数据，不能进入 PyTorch/model 主表。
- `LOCAL_SHM` 当前 1proc 启动失败，必须作为失败记录保留，不能填吞吐。

## 主实验矩阵

### E1：单 GPU 端到端主表

目的：建立 RecStore 与 TorchRec 的投稿主表。

固定项：

- 数据：Criteo day 0 切片。
- 模型：`model_zoo/rs_demo/run_mock_stress.py` 当前 DLRM-like runner。
- 指标：`mean_step_total_ms`、`p95_step_total_ms`、`samples_per_sec`、`lookup_mrows_per_sec`、`update_mrows_per_sec`。
- warmup：至少 5 step，不进入统计。
- repeat：投稿版至少 3 次，报告 mean、median、std、CV。

维度：

- `data_rows`: `131072,524288`，可追加 `1048576`。
- `batch_size`: `512,1024,2048,4096`。
- `num_embeddings`: `200000,800000,2000000,4000000`。
- `embedding_dim`: `64,128`。
- lanes：`torchrec-hbm-1p`、`torchrec-uvm-1p`、`recstore-brpc-pet-1p`、`recstore-brpc-eh-1p`、`recstore-grpc-pet-1p`、`recstore-brpc-map-1p`。

推荐命令：

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile stress \
  --output-root /nas/home/shq/docker/rs_demo/paper_e2e_submit_single_gpu_r3 \
  --data-rows 131072,524288 \
  --batch-sizes 512,1024,2048,4096 \
  --num-embeddings 200000,800000,2000000,4000000 \
  --embedding-dims 64,128 \
  --steps 60 \
  --warmup-steps 5 \
  --repeat 3 \
  --include-ablation-lanes \
  --only-lanes torchrec-hbm-1p,torchrec-uvm-1p,recstore-brpc-pet-1p,recstore-brpc-eh-1p,recstore-grpc-pet-1p,recstore-brpc-map-1p \
  --skip-rdma-ps
```

### E2：容量压力

目的：回答 TorchRec-HBM/UVMCache 与 RecStore 在大 embedding table 下的容量敏感性。

维度：

- `num_embeddings`: `800000,2000000,4000000,8000000`。
- `embedding_dim`: `128` 为主，必要时追加 `64`。
- `batch_size`: `1024,4096`。
- lanes：`torchrec-hbm-1p`、`torchrec-uvm-1p`、最佳 RecStore lane 集合。

注意：

- 如果 TorchRec-HBM 没有 OOM，不能按理论 embedding size 声称 OOM。
- 需要记录实际 GPU memory、UVM 行为、runner 是否 lazy/materialized embedding。
- 如果触发 OOM，记录错误日志路径和退出码，OOM 是结果的一部分。

推荐命令：

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile stress \
  --output-root /nas/home/shq/docker/rs_demo/paper_e2e_submit_capacity_8m \
  --data-rows 524288 \
  --batch-sizes 1024,4096 \
  --num-embeddings 800000,2000000,4000000,8000000 \
  --embedding-dims 128 \
  --steps 60 \
  --warmup-steps 5 \
  --repeat 3 \
  --include-ablation-lanes \
  --only-lanes torchrec-hbm-1p,torchrec-uvm-1p,recstore-brpc-pet-1p,recstore-brpc-eh-1p,recstore-grpc-pet-1p,recstore-brpc-map-1p \
  --skip-rdma-ps
```

### E3：单机多 GPU 扩展性

目的：回答多 GPU 训练时 RecStore 是否仍能保持端到端优势。

硬件要求：

- 至少 2 张 GPU；投稿建议 2/4/8 GPU。
- 相同 GPU 型号、CPU、内存、驱动和 PyTorch/TorchRec 版本。

维度：

- `nproc_per_node`: `1,2,4,8`，按机器实际 GPU 数递增。
- `batch_size`: 每 GPU 固定 batch 和全局 batch 固定两种策略至少选择一种，并在论文中说明。
- `num_embeddings`: `800000,2000000`。
- lanes：`torchrec-hbm-*p`、`torchrec-uvm-*p`、`recstore-brpc-pet-*p`、`recstore-local-shm-pet-*p`。

当前脚本已有 2proc lane，但还需要在多 GPU 节点扩展到 4/8proc lane 后再跑。未补齐前，论文不能声称多 GPU scalability。

建议命令：

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile stress \
  --output-root /nas/home/shq/docker/rs_demo/paper_e2e_submit_multigpu_2p \
  --data-rows 524288 \
  --batch-sizes 1024,4096 \
  --num-embeddings 800000,2000000 \
  --embedding-dims 64,128 \
  --steps 60 \
  --warmup-steps 5 \
  --repeat 3 \
  --only-lanes torchrec-hbm-2p,recstore-brpc-pet-2p,recstore-local-shm-pet-2p \
  --skip-rdma-ps
```

### E4：RecStore 后端与传输消融

目的：解释端到端优势来自哪里，避免只报告 best lane。

维度：

- 传输：`BRPC`、`GRPC`、`LOCAL_SHM`。
- 索引：`DRAM_PET_HASH`、`DRAM_EXTENDIBLE_HASH`、`DRAM_UNORDERED_MAP`。
- prefetch：`0,1,4,8`。
- `ps_kv_backend`: 当前主表固定 `recstore_dram`；如果新增 SSD/backend integration，必须作为新 lane 单独标注。

必须报告：

- best RecStore lane。
- 每个消融 lane 相对 best non-prefetch RecStore 的保留率。
- failure/timeout/startup hang 的聚合原因。

推荐命令：

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile stress \
  --output-root /nas/home/shq/docker/rs_demo/paper_e2e_submit_ablation \
  --data-rows 131072,524288 \
  --batch-sizes 512,1024,4096 \
  --num-embeddings 200000,800000 \
  --embedding-dims 64,128 \
  --steps 60 \
  --warmup-steps 5 \
  --repeat 3 \
  --include-ablation-lanes \
  --only-lanes recstore-brpc-pet-1p,recstore-brpc-eh-1p,recstore-brpc-map-1p,recstore-grpc-pet-1p,recstore-grpc-eh-1p,recstore-brpc-pet-prefetch1-1p,recstore-brpc-pet-prefetch4-1p,recstore-brpc-pet-prefetch8-1p \
  --skip-rdma-ps
```

### E5：RDMA PS/network 校准

目的：证明 RDMA 参数服务器路径本身可用，并量化网络层上限。

维度：

- backend/index：`DRAM_PET_HASH`、`DRAM_EXTENDIBLE_HASH`。
- value size：`256,512,1024` bytes。
- batch keys：`64,128,500,1024`。
- client processes：`1,2,4,6,8`。
- repeat：至少 3。

报告方式：

- 主指标：`M keys/s`，按 `run/fetch` phase 报 median、p95、CV。
- 与 PyTorch/model 分开成独立小节。
- 只能作为 RecStore transport calibration，不能写成 RecStore 端到端 RDMA 已加速 TorchRec。

### E6：分解与稳定性

目的：把端到端胜负拆成可解释项。

每个关键配置必须给出：

- `step_total_ms`。
- `embed_lookup_local_ms`。
- `sparse_update_ms`。
- `lookup_mrows_per_sec`。
- `update_mrows_per_sec`。
- repeat mean/median/std/CV。
- failed/skipped count。

建议主文放 3 个代表配置：

- small batch win：`batch_size=512/1024`，RecStore 明显优于 TorchRec-UVMCache。
- large batch loss：`batch_size=4096`，TorchRec-UVMCache 更强。
- large capacity：`num_embeddings=2000000/4000000/8000000`，展示容量趋势。

完整矩阵放 appendix 或 artifact。

## 报告聚合要求

投稿版 LaTeX/CSV 需要新增或确认以下聚合：

- `summary_gap.csv` 中使用 repeat 聚合后的 median，而不是 first matching row。
- failure table 按 `(lane,status,reason)` 聚合，不逐行铺开 16 条相同 `LOCAL_SHM` 失败。
- 对每个配置输出 `best_recstore_label`，但主文同时保留固定 lane 表，避免只挑 best 导致选择偏差。
- 输出 batch 分组的几何均值：`batch<=1024`、`batch=2048`、`batch>=4096`。
- 输出容量分组的几何均值：`num_embeddings<=800000`、`2M`、`4M+`。
- 明确 `skipped` 不进入几何均值，`failed/OOM` 单独计数。

## 投稿表述红线

不能写：

- “RecStore RDMA 端到端快于 TorchRec”，除非 PyTorch/model 层已经接入 RDMA lane 并跑通。
- “TorchRec-HBM 在 4M embeddings OOM”，除非日志真实出现 OOM。
- “单机多卡可扩展”，除非在多 GPU 节点补跑成功。
- “LOCAL_SHM 更快”，当前 1proc LOCAL_SHM 是失败项。
- “存储层优势解释端到端优势”，除非同配置有 PS/network 和 PyTorch/model retention 分析。

可以写：

- “在当前单 GPU PyTorch/model 层，RecStore best lane 在多数小 batch 配置上优于 TorchRec-HBM/UVMCache。”
- “大 batch 下 TorchRec-UVMCache 的 GPU 批处理优势明显，RecStore 需要进一步优化 lookup/update 调度。”
- “RDMA PS/network 层已经校准到独立吞吐结果，但尚未作为 PyTorch/model 主路径。”

## 下一步优先级

1. 在当前单 GPU 机器补 `repeat=3` 的容量和 batch=2048 缺口。
2. 修改报告聚合，使用 repeat median 和聚合 failure table。
3. 找 2/4/8 GPU 节点补跑 E3。
4. 若论文要主打 RDMA，先实现并验证 PyTorch/model RDMA lane，再把 RDMA 纳入主表。
5. 安装 LaTeX 编译器或换环境生成 PDF artifact。
