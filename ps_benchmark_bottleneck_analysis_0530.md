# PS Benchmark RDMA 瓶颈分析

## 结论

当前 PS benchmark 的主要问题不是一句“server 慢”能解释完。更准确的结论是：

1. **原始 PS benchmark 的 RDMA fetch 热路径和 RDMA RC 专项 benchmark 没有对齐**，adapter/prefetch 调度本身先把吞吐上限压低。
2. 目前已经逐步去掉 adapter buffer 分配/清零、client response 中间拷贝，以及一部分 DRAM fixed-row 读取开销。此后，**server GET payload copy / batch get / 单 polling worker 处理能力成为当前主瓶颈**。
3. 原始 PS benchmark 的容量配置没有发现 KVEngine YCSB 那种 DRAM allocator 容量不足问题；这次瓶颈不是因为 load 没装进去或大量 missing rows。

因此，现在不再适合笼统说“server 慢”，但下一步最值得继续攻的是 **server 真实 payload 路径**，尤其是 `PetPSServer::HandleGet -> CachePS::GetParameterFlat -> KVEngineComposite::BatchGetFlat -> ValueStore` 这条链路。

## 实验依据

同一组核心参数：

- `server_shards=1`
- `client_processes_per_ip=4`
- `client_threads_per_process=1`
- `batch_keys=500`
- `value_size=512`
- `prefetch_depth=16`
- `rdma_rc_qps_per_client_per_shard=16`
- `rdma_rc_slots_per_qp=1`

历史/本轮关键结果：

| 场景 | 聚合吞吐 | 含义 |
|---|---:|---|
| PS default | 约 `2.82M keys/s` | 原始 PS RDMA fetch 路径 |
| PS `status_only` | 约 `3.20M keys/s` | 去掉 server payload 后，原 adapter 路径仍然低 |
| PS `skip_client_copy` | 约 `3.27M keys/s` | 只跳过 PetPS client copy，提升有限 |
| PS default + `server-rdma-threads=16` | 约 `2.26M keys/s` | 增加 server polling threads 没解决调度瓶颈 |
| RDMA RC async_stream，4 client、1 thread | 约 `3.75M keys/s` | 单线程 RC 专项与 PS default 同量级 |
| RDMA RC async_stream，4 client、16 thread | 约 `43.09M keys/s` | 专项 benchmark 在高并发下能冲高 |
| PS direct async + `status_only` + `skip_load` | 约 `16.1M keys/s` | 绕过 adapter 后，PS benchmark 调度上限明显提高 |
| PS default + adapter buffer pool | 约 `4.40M keys/s` | 去掉每 prefetch 的 receive buffer 分配/整块清零后，真实 payload 提升约 56% |
| PS `status_only` + adapter buffer pool | 约 `15.86M keys/s` | 去掉 server payload 后，adapter 路径已接近 direct async status-only |
| PS default + `resize+memcpy` result copy | 约 `4.72M keys/s` | `GetPrefetchResultFlat` 不再用 `vector::assign`，收益有限 |
| PS `skip_prefetch_result_copy` 诊断 | 约 `4.60M keys/s` | 跳过 adapter result vector materialize 没有明显提升，说明它不是当前主墙 |
| PS `index_only` 诊断 | 约 `12.09M keys/s` | server 只做 index lookup，不复制 value，`handle_get_avg_ns` 约 `38us/batch` |
| PS `skip_client_copy` 诊断 | 约 `4.85M keys/s` | 跳过 `PetPSClient::WaitRPCFinish` response copy，收益小，server row copy 仍重 |
| PS default + DRAM fixed-row flat read | 约 `4.94M keys/s` | `get_row_copy_avg_ns` 从约 `59us/batch` 降到约 `51us/batch` |
| PS default + borrowed RC response payload | 约 `5.20M keys/s` | client `copy_response_avg_ns=0`，但 server `handle_get_avg_ns` 仍约 `92us/batch` |
| PS borrowed response + `skip_prefetch_result_copy` 诊断 | 约 `5.25M keys/s` | 最后一跳 vector copy 不是主要剩余瓶颈 |

最关键的是 `PS direct async + status_only`：它仍然通过 PetPS server 和 RDMA RC transport，但绕过了 `RDMAPSClientAdapter` 的 prefetch map、buffer memset 和 result assign。吞吐从原 PS `status_only` 的约 `3.2M` 到约 `16.1M keys/s`，说明 **adapter/benchmark 调度路径是第一层大瓶颈**。

本轮 P0 先处理了其中最大的一块：`PrefetchParameter` 不再每次通过 `PetPSClient::GetReceiveBuffer` 申请/清零 response buffer，也不再对 `500 * 512B` 的 payload 区域整块 `memset`。这个改动后，`status_only` 从约 `3.20M keys/s` 提升到约 `15.86M keys/s`，基本追上 direct async 的 status-only 诊断结果；真实 payload 从约 `2.82M keys/s` 提升到约 `4.40M keys/s`。

后续又做了三步：

- `GetPrefetchResultFlat` 从 `vector::assign` 改成 `resize + memcpy`，真实 payload 到约 `4.72M keys/s`。
- `DRAM_VALUE_STORE` 增加 fixed-row flat read 快路径，真实 payload 到约 `4.94M keys/s`，server `get_row_copy_avg_ns` 降到约 `51us/batch`。
- 单 shard prefetch 借用 RC response slot payload，跳过 `PetPSClient::WaitRPCFinish` 的中间 payload copy，真实 payload 到约 `5.20M keys/s`，client `copy_response_avg_ns=0`。

这些优化都有效，但没有量级变化。与 `status_only` 的约 `15.86M keys/s`、`index_only` 的约 `12.09M keys/s` 相比，当前主墙已经非常明确：**真实 value payload copy 和 server 单 polling worker 的 batch 处理能力**。

## 最大问题 1：PS adapter prefetch 路径太重

相关代码：

- `src/ps/rdma/rdma_ps_client_adapter.cc`
- `src/ps/rdma/rdma_ps_client_adapter.h`

关键函数：

- `RDMAPSClientAdapter::PrefetchParameter`
- `RDMAPSClientAdapter::GetPrefetchResultFlat`
- `RDMAPSClientAdapter::WaitForPrefetch`
- `RDMAPSClientAdapter::MarkPrefetchConsumed`
- `RDMAPSClientAdapter::GetPrefetchState`

优化前 `PrefetchParameter` 每个 fetch batch 都会做这些事情：

```cpp
const std::size_t response_bytes =
    petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
float* buffer = static_cast<float*>(client_->GetReceiveBuffer(response_bytes));
std::memset(buffer, 0, response_bytes);

const int rpc_id = SubmitGetParameter(keys, buffer, true, 0);

std::lock_guard<std::mutex> guard(state_mu_);
const uint64_t prefetch_id = next_prefetch_id_++;
prefetches_.emplace(prefetch_id, PrefetchState{buffer, rpc_id, ...});
```

对 `batch_keys=500`、`value_size=512` 来说，单次 response 约 `256KB`。也就是说，每个 batch 提交前都会做一大块 receive buffer 申请和清零。

早期 `GetPrefetchResultFlat` 还会做：

```cpp
WaitForPrefetch(prefetch_id);
values->assign(state.buffer, state.buffer + value_count);
RevokeRPCResource(state.rpc_id);
MarkPrefetchConsumed(prefetch_id);
```

这里的问题有三类：

- 每请求 `memset` 大 buffer。
- 每请求进入 `prefetches_` 哈希表，并受 `state_mu_` 保护。
- 每请求把 RDMA receive buffer 再 `assign` 到 `std::vector<float>`，多一次分配/拷贝语义。

这和 RDMA RC 专项 benchmark 的 async stream 路径不一致。

本轮已改成 adapter 自己维护可复用 prefetch buffer pool：

```cpp
float* buffer = AcquirePrefetchBuffer(response_bytes, &buffer_id);
auto* status_word = petps::FixedSlotStatusWord(
    buffer, static_cast<std::size_t>(keys.Size()), FLAGS_value_size);
*status_word = static_cast<std::int32_t>(petps::RpcStatus::kPending);

int rpc_id = -1;
try {
  rpc_id = SubmitGetParameter(keys, buffer, true, 0);
} catch (...) {
  ReleasePrefetchBuffer(buffer_id);
  throw;
}
```

对应函数：

- `RDMAPSClientAdapter::AcquirePrefetchBuffer`
- `RDMAPSClientAdapter::ReleasePrefetchBuffer`
- `RDMAPSClientAdapter::MarkPrefetchConsumed`
- `RDMAPSClientAdapter::PrefetchParameter`

这一步解决的是最粗的 per-request 分配/清零问题。后续又做了两个 client/adapter 侧优化：

- `RDMAPSClientAdapter::GetPrefetchResultFlat` 从 `values->assign(...)` 改为 `values->resize(...) + std::memcpy(...)`，避免 `assign` 的额外语义开销。
- `PetPSClient::BorrowGetResultPayload` + `RDMAPSClientAdapter::BorrowPrefetchResult` 让单 shard prefetch 在消费时直接从 RC response slot payload materialize 结果，跳过 `PetPSClient::WaitRPCFinish` 中的中间 response copy；slot 仍在 `GetPrefetchResultFlat` 消费完成后 `RevokeRPCResource`。

现在剩下还没有完全对齐 RC 专项 benchmark 的地方是：

- `GetPrefetchResultFlat` 仍然会把结果 materialize 到 `std::vector<float>`，这是 generic PS API 语义；但 skip-copy 诊断只从约 `5.20M` 到约 `5.25M keys/s`，不是当前主瓶颈。
- `prefetches_` 仍然是 `unordered_map + state_mu_`。
- benchmark 的 generic PS API 仍然是 prefetch id 语义，不是 RC 专项那种固定 slot/ring output buffer。

## 最大问题 2：PS benchmark 没有复用 RC 专项的 direct async 调度模式

相关代码：

- `src/benchmark/ps_transport_benchmark.cc`
- `src/benchmark/rdma_rc_transport_benchmark.cc`

PS 原始 fetch transaction 路径：

- `RunPrefetchFetchTransactions`
- `PrefetchFlat`
- `ConsumePrefetchFlat`
- `BasePSClient::PrefetchParameter`
- `BasePSClient::GetPrefetchResultFlat`

RC 专项 async stream 路径：

- `RunAsyncStreamOperation`
- `BaseParameterClient::GetParameter(..., isAsync=true, ...)`
- `BaseParameterClient::WaitRPCFinish`
- `BaseParameterClient::RevokeRPCResource`

RC 专项 benchmark 的核心调度方式是：

```cpp
std::vector<std::vector<float>> outputs;
for (int i = 0; i < FLAGS_async_depth; ++i) {
  outputs.emplace_back(output_size, 0.0f);
}

rpc_ids[slot] = client->GetParameter(
    input.key_array,
    outputs[slot].data(),
    true,
    0);

client->WaitRPCFinish(rpc_id);
client->RevokeRPCResource(rpc_id);
```

它的特点是：

- output buffers 预分配。
- 不走 `RDMAPSClientAdapter::PrefetchParameter`。
- 不走 `prefetches_` map。
- 不做 `GetPrefetchResultFlat` 的 `vector::assign`。
- request/revoke 直接围绕 PetPS RC slot 执行。

这就是为什么 RC 专项能在更高并发下跑到几十 M keys/s，而 PS benchmark 原路径卡在几 M keys/s。

## 最大问题 3：server payload 路径仍然很重

相关代码：

- `src/ps/rdma/petps_server.cc`
- `src/ps/rdma/petps_client.cc`

server 关键逻辑：

- `PetPSServer::HandleGet`
- `PetPSServer::PollingThreadMain`
- GET 响应完成路径中的 response payload/status 写入

client 等待和复制逻辑：

- `PetPSClient::WaitRPCFinish`
- `PetPSClient::RevokeRPCResource`
- `PetPSClient::GetParameter`

真实 payload 模式下，server profile 里能看到：

- `handle_get_avg_ns` 约 `90us - 135us` 级别，视 server threads、数据命中、payload 模式变化。
- `get_row_copy_avg_ns` 是 `handle_get_avg_ns` 中的大头之一。
- `get_missing_rows=0` 的原始 PS runs 表示数据命中，不是容量不足导致的 missing。

在 direct async + `skip_load` 但不 `status_only` 的诊断里，吞吐只有约 `0.97M keys/s`，原因是没有 preload，server 走了 100% missing rows 的 zero-fill/row-copy 路径。这条不能当真实 payload 吞吐结论，但能证明 server payload/zero-fill 本身很重。

在 direct async + `status_only` 下，server payload 几乎被去掉：

- server `handle_get_avg_ns` 约 `80ns - 90ns`
- client `pending_rpc_peak=16`
- 聚合吞吐约 `16.1M keys/s`

这说明绕过 adapter 后，纯调度上限高很多；而真实 payload 模式下，server GET payload copy 会成为下一层瓶颈。

本轮 adapter buffer pool 后，这个判断更明确：

- `status_only + adapter buffer pool` 聚合约 `15.86M keys/s`。
- `default payload + adapter buffer pool` 聚合约 `4.40M keys/s`。
- 两者同样 `prefetch_depth=16`，client profile 都能看到 `pending_rpc_peak=16`，说明这组参数下 in-flight 深度已经打满，不是“完全没并发”。
- 真实 payload 模式下 server profile 中 `handle_get_avg_ns` 约 `108us - 111us`，`get_row_copy_avg_ns` 约 `62us - 64us`，单 server polling thread 理论上也就在约 `9K batches/s` 这一档，乘以 `500 keys/batch` 正好接近 `4.4M keys/s`。
- P3 borrowed response 后，client 侧 `copy_response_avg_ns=0`，但真实 payload 仍只有约 `5.20M keys/s`；同参数下 server `handle_get_avg_ns` 约 `92us/batch`，`get_row_copy_avg_ns` 约 `49us/batch`。
- `skip_prefetch_result_copy` 诊断只到约 `5.25M keys/s`，说明最后的 vector materialize 已经不是主要剩余瓶颈。

所以当前更精确的瓶颈排序已经变成：

1. **已缓解：adapter 每 prefetch 分配 receive buffer + 整块 memset。**
2. **已缓解：client `WaitRPCFinish` 中间 response copy。**
3. **已缓解一部分：DRAM fixed-row flat read 的 per-row value access/copy 开销。**
4. **当前主瓶颈：server GET payload copy / batch get / single polling thread 处理能力。**

## 为什么不能只说 server 是问题

如果 server 是唯一瓶颈，那么去掉 server payload 的 `status_only` 应该能直接接近 RDMA RC 专项 benchmark。但实际不是：

- 原 PS `status_only` 只有约 `3.2M keys/s`。
- direct async `status_only` 能到约 `16.1M keys/s`。

两者 server 都是轻 payload/status-only，但吞吐相差约 5 倍。这说明原 PS adapter/benchmark 调度路径本身就很重。

同时，direct async `status_only` 仍低于 RC 专项约 `43M keys/s`，所以也不能反过来说 server 没问题。更合理的分层判断是：

1. **原始 PS benchmark 的第一瓶颈：client adapter/prefetch/result 调度路径。**
2. **当前已经优化掉 adapter/client 的几块大开销，下一瓶颈：server polling/dispatch 和 GET payload copy。**

## 容量问题检查

参考文档：

- `kvengine_ycsb_dram_capacity_note.md`

KVEngine YCSB 那次的问题是 `ConcurrentSlabMemoryPool` 容量估算没有考虑：

- 1MB chunk 切分。
- chunk header。
- bitmap 元数据。
- entry 数向下对齐到 64。

这导致 10M records 的 DRAM value store 在 load 阶段 OOM。

本次 PS benchmark 原始结果没有同类问题：

- 配置路径：`results/benchmark_ps_sched_0529/default_1s4c_t1/configs/rdma_repeat_0.json`
- `record_count=200000`
- `value_size=512`
- `dram_allocator=PERSIST_LOOP_SLAB`
- `capacity_bytes=125829120`

`PERSIST_LOOP_SLAB` 对应：

- `src/memory/allocators/persist_loop_slab_allocator.h`
- `src/memory/allocators/persist_loop_slab_allocator.cc`
- `PersistLoopShmMalloc`

它不是 `ConcurrentSlabMemoryPool` 的 chunk slab 模型，而是 8B block + bitmap 的 first-fit 分配。200k 条 512B value 约需要：

```text
200000 * (512 + 8) ~= 104,000,000 bytes
```

当前配置给了 `125829120 bytes`，有余量。原始 PS server profile 的 run 阶段也显示 `get_missing_rows=0`，说明不是容量不足导致的读缺失。

## 优化建议优先级

### P0：优化 PS RDMA fetch adapter 热路径

目标：让 PS benchmark fetch 调度更接近 RC async stream。

已完成：

- `RDMAPSClientAdapter` 增加 adapter-owned prefetch buffer pool。
- `PrefetchParameter` 只初始化 status word，不再整块 `memset` response buffer。
- `MarkPrefetchConsumed` 将 buffer id 放回 free list。
- `PrefetchParameter` submit 异常时调用 `ReleasePrefetchBuffer`，避免失败路径泄漏 buffer slot。
- `ps_transport_benchmark` 和 `run_benchmark_ps.py` 增加 `prefetch_depth <= rdma_rc_qps_per_client_per_shard * rdma_rc_slots_per_qp` 校验，避免 benchmark 默认 depth 16 但 slot 只有 4 时在 `PetPSClient::AcquireIdleSlot` 里 abort。

仍建议：

- 避免 `GetPrefetchResultFlat` 对结果做 `vector::assign`。
- 将 prefetch state 从全局 `unordered_map + mutex` 改成固定 ring/slot id，至少 benchmark 路径可以这样做。
- 对单 shard RDMA fetch 提供 direct async fast path，保留默认语义但给 benchmark/训练路径可选优化。

受影响函数：

- `RDMAPSClientAdapter::PrefetchParameter`
- `RDMAPSClientAdapter::GetPrefetchResultFlat`
- `RDMAPSClientAdapter::WaitForPrefetch`
- `RDMAPSClientAdapter::RevokeRPCResource`
- `RDMAPSClientAdapter::AcquirePrefetchBuffer`
- `RDMAPSClientAdapter::ReleasePrefetchBuffer`
- `PetPSClient::AcquireIdleSlot`

本轮验证：

- `cmake --build build --target test_rdmaps_client_adapter test_ps_transport_benchmark ps_transport_benchmark -j`
- `ctest --test-dir build -R 'test_rdmaps_client_adapter|test_ps_transport_benchmark' -VV`
- `python3 -m unittest src/test/scripts/test_run_benchmark_ps.py`
- RDMA smoke：`results/rdma_ps_adapter_buffer_smoke_0530_depth4/summary.csv`
- RDMA default payload：`results/benchmark_ps_p0_adapter_buffer_0530/default_1s4c_t1/summary.csv`
- RDMA status-only：`results/benchmark_ps_p0_adapter_buffer_0530/status_only_1s4c_t1/summary.csv`

### P1：优化 server GET payload copy

目标：降低真实 payload 模式下 `handle_get_avg_ns` 和 `get_row_copy_avg_ns`。

已完成：

- 新增 `BaseKV::BatchGetIndexOnly` / `KVEngineComposite::BatchGetIndexOnly` / `CachePS::ProbeParameterIndex`，并通过 `--rdma-rc-fake-get-mode index_only` 拆出 index lookup 成本。
- 新增 `ValueStore::ReadFlatFixedRows`，`DramValueStore` 对固定 row size 场景覆盖实现。
- `KVEngineComposite::BatchGetFlat` 在 `default_value_size_hint == row_bytes` 时走 DRAM fixed-row flat read 快路径。

结论：

- `index_only` 约 `12.09M keys/s`，`handle_get_avg_ns` 约 `38us/batch`，index lookup 不是最大瓶颈。
- DRAM fixed-row flat read 将真实 payload 从约 `4.72M` 提升到约 `4.94M keys/s`，`get_row_copy_avg_ns` 约 `59us -> 51us/batch`，有收益但不够。
- 真实 payload 仍显著低于 `status_only` 和 `index_only`，说明 value payload copy/response payload 生成仍是主墙。

仍建议：

- 继续 profile `PetPSServer::HandleGet` 内部 batch get、row copy、response write。
- 检查是否能减少 per-row `DirectPtr/Ptr + memcpy`，或对固定 512B row 做更低开销的批量拷贝。
- 区分 zero-fill、hit copy、status-only、index-only 四种路径，不要混在一个结论里。

受影响函数：

- `PetPSServer::HandleGet`
- `PetPSClient::WaitRPCFinish`
- `PetPSClient::GetParameter`
- `CachePS::GetParameterFlat`
- `CachePS::ProbeParameterIndex`
- `KVEngineComposite::BatchGetFlat`
- `KVEngineComposite::BatchGetIndexOnly`
- `DramValueStore::ReadFlatFixedRows`

### P2：去掉 client response 中间 copy

目标：让 PS prefetch 消费路径更接近 RC slot/ring 语义。

已完成：

- `PetPSClient::BorrowGetResultPayload`：等待指定 RPC 完成后，返回 RC response slot payload 指针，不把 payload memcpy 到 adapter buffer。
- `RDMAPSClientAdapter::BorrowPrefetchResult`：单 shard prefetch 在 `GetPrefetchResultFlat` 中借用 RC response payload。
- slot 生命周期仍由 `RevokeRPCResource` 控制，结果 materialize 完成后释放。

结果：

- P3 默认真实 payload 约 `5.20M keys/s`。
- client profile 中 `copy_response_avg_ns=0`。
- P3 + `--rdma-adapter-skip-prefetch-result-copy` 约 `5.25M keys/s`，说明最后 vector copy 不是主要剩余瓶颈。

受影响函数：

- `PetPSClient::BorrowGetResultPayload`
- `RDMAPSClientAdapter::BorrowPrefetchResult`
- `RDMAPSClientAdapter::GetPrefetchResultFlat`

### P3：重新设计 server polling 多线程扩展

目标：让 `server-rdma-threads` 增加时真正提升吞吐，而不是空扫/竞争。

当前观察：

- `server-rdma-threads=16` 没有提升原 PS default，反而下降。
- profile 里多线程时 `scan_rounds` 很高、`scan_hit_pct` 很低，说明 polling 扩展方式不是简单加线程就能变快。

建议：

- 按 client id / QP / shard 对 polling ownership 做明确分片。
- 避免多个 polling thread 扫同一批 mostly-empty slots。
- 将 ready queue 或 completion handoff 做成更直接的调度结构。

受影响函数：

- `PetPSServer::PollingThreadMain`
- `RcShardServerTransport` 相关 slot 扫描/完成逻辑

## 后续 benchmark 口径

后续报告时建议至少分这几类结果，不要混在一起：

1. `PS default fetch`：真实 PS adapter 路径。
2. `PS status_only`：原 adapter 调度 + server payload 去掉。
3. `PS direct async status_only`：绕过 adapter 后的调度上限。
4. `PS direct async payload`：绕过 adapter 后的真实 payload。
5. `RDMA RC async_stream`：RC 专项，不等同 PS/network。

这样才能分别回答：

- adapter 调度占多少？
- server payload 占多少？
- polling 并发是否有效？
- PS/network 和 RC transport 专项之间还差在哪？
