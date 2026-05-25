# Composite KV 开发指南

`KVEngineComposite` 把 KV 拆成三个层次：`Index` 保存 key 到 handle 的映射，`ValueStore` 管理 handle 指向的字节，`IOBackend` 只处理页级 SSD IO。开发新的本地 KV 能力时，优先扩展其中一个层次。

## 组件组合

| 组件 | 实现 |
|------|----------|
| `Index` | `DRAM_EXTENDIBLE_HASH`、`DRAM_UNORDERED_MAP`、`DRAM_PET_HASH`、`SSD`、`SSD_EXTENDIBLE_HASH` |
| `ValueStore` | `DRAM_VALUE_STORE`、`SSD_VALUE_STORE`、`TIERED_VALUE_STORE` |
| `IOBackend` | `IOURING`、`SPDK` |
| DRAM allocator | `PERSIST_LOOP_SLAB`、`R2_SLAB` |
| SSD allocator | `SSD_SLAB`、`SSD_BUDDY` |

`ResolveEngine` 限制以下组合：

| 组合 | 状态 |
|------|------|
| DRAM index + `DRAM_VALUE_STORE` | 支持 |
| DRAM index + `SSD_VALUE_STORE` | 支持 |
| DRAM index + `TIERED_VALUE_STORE` | 支持 |
| SSD index + `SSD_VALUE_STORE` | 支持 |
| SSD index + `DRAM_VALUE_STORE` | 拒绝 |
| SSD index + `TIERED_VALUE_STORE` | 拒绝 |

显式外部引擎不参与上表的 `KVEngineComposite` 组合规则。`KVEngineFasterKV` 使用 FasterKV 内部的内存 hash index；它的可选 `fasterkv.storage` 只控制 FasterKV hybrid log/value backend：`memory` 使用 `NullDisk`，`ssd` 使用 `FileSystemDisk` 并可将 hybrid log/value 放到 `fasterkv.log_path`。这和 RecStore 内置的 `index` / `value` 分层不是同一套配置面。

## 配置样例

??? example "DRAM index + DRAM value"

    ```json
    {
      "capacity": 1000000,
      "index": {
        "type": "DRAM_EXTENDIBLE_HASH"
      },
      "value": {
        "type": "DRAM_VALUE_STORE",
        "default_value_size_hint": 512,
        "path": "/dev/shm/recstore_data/value",
        "dram_allocator": {
          "type": "PERSIST_LOOP_SLAB",
          "capacity_bytes": 512000000
        }
      }
    }
    ```

??? example "DRAM index + SSD value"

    ```json
    {
      "capacity": 1000000,
      "index": {
        "type": "DRAM_UNORDERED_MAP"
      },
      "value": {
        "type": "SSD_VALUE_STORE",
        "default_value_size_hint": 512,
        "path": "/data/recstore/value_pages.db",
        "ssd_allocator": {
          "type": "SSD_BUDDY",
          "capacity_bytes": 512000000,
          "min_block_size": 128,
          "max_block_size": 65536,
          "io": {
            "type": "IOURING",
            "queue_depth": 512,
            "base_offset_bytes": 4096
          }
        }
      }
    }
    ```

??? example "DRAM index + tiered value"

    ```json
    {
      "capacity": 1000000,
      "index": {
        "type": "DRAM_EXTENDIBLE_HASH"
      },
      "value": {
        "type": "TIERED_VALUE_STORE",
        "default_value_size_hint": 512,
        "dram_allocator": {
          "type": "PERSIST_LOOP_SLAB",
          "capacity_bytes": 128000000,
          "path": "/dev/shm/recstore_data/tiered_dram"
        },
        "ssd_allocator": {
          "type": "SSD_BUDDY",
          "capacity_bytes": 512000000,
          "path": "/data/recstore/tiered_value_pages.db",
          "min_block_size": 128,
          "max_block_size": 65536,
          "io": {
            "type": "IOURING",
            "queue_depth": 512,
            "base_offset_bytes": 4096
          }
        },
        "tiering": {
          "high_watermark_ratio": 0.85
        }
      }
    }
    ```

??? example "SSD index + SSD value"

    ```json
    {
      "capacity": 1000000,
      "index": {
        "type": "SSD_EXTENDIBLE_HASH",
        "path": "/data/recstore/index_pages.db",
        "io": {
          "type": "IOURING",
          "queue_depth": 512,
          "base_offset_bytes": 0
        }
      },
      "value": {
        "type": "SSD_VALUE_STORE",
        "default_value_size_hint": 512,
        "path": "/data/recstore/value_pages.db",
        "ssd_allocator": {
          "type": "SSD_SLAB",
          "capacity_bytes": 512000000,
          "size_classes": [128, 256, 512, 1024, 4096],
          "io": {
            "type": "IOURING",
            "queue_depth": 512,
            "base_offset_bytes": 4096
          }
        }
      }
    }
    ```

??? example "完整配置示例"

    ```json
    {
      "cache_ps": {
        "num_threads": 32,
        "base_kv_config": {
          "capacity": 1000000,
          "index": {
            "type": "SSD_EXTENDIBLE_HASH",
            "path": "/data/recstore/index_pages.db",
            "io": {
              "type": "IOURING",
              "queue_depth": 512,
              "base_offset_bytes": 0
            }
          },
          "value": {
            "type": "SSD_VALUE_STORE",
            "default_value_size_hint": 512,
            "path": "/data/recstore/value_pages.db",
            "ssd_allocator": {
              "type": "SSD_BUDDY",
              "capacity_bytes": 512000000,
              "min_block_size": 128,
              "max_block_size": 65536,
              "io": {
                "type": "IOURING",
                "queue_depth": 512,
                "base_offset_bytes": 4096
              }
            }
          }
        }
      }
    }
    ```

## Index 接口

`Index` 只保存 `Key_t -> Value_t`。value 字节由 `ValueStore` 读写。

实现新索引时需要覆盖：

```cpp
void Get(Key_t key, Value_t& pointer, unsigned tid) override;
void Put(Key_t key, Value_t pointer, unsigned tid) override;
void BatchGet(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) override;
void BatchPut(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) override;
bool Delete(Key_t& key) override;
```

未命中必须返回 `kValueHandleNone`。注册示例：

```cpp
FACTORY_REGISTER(Index, MY_INDEX, MyIndex, const BaseKVConfig&);
```

还要在 `engine_selector.h` 中加入 `MY_INDEX`，写清楚它需要哪些配置。SSD index 要求 `index.path` 和 `index.io`。

## ValueStore 接口

`ValueStore` 负责分配、读写、释放 handle。`KVEngineComposite` 根据 `SlotCapacity(handle)` 判断覆盖还是重新分配。

核心接口：

```cpp
uint64_t Alloc(size_t size) override;
void Write(uint64_t handle, const void* data, size_t size) override;
uint64_t AllocAndWrite(const void* data, size_t size) override;
size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override;
void Free(uint64_t handle) override;
size_t SlotCapacity(uint64_t handle) const override;
```

DRAM value store 可以实现 `DirectPtr`，让 `BatchGet` 少一次拷贝。SSD value store 不应返回直接指针。

`TIERED_VALUE_STORE` 按容量水位选择 DRAM 或 SSD：DRAM 预留空间未超过 `high_watermark_ratio` 时分配 DRAM，否则分配 SSD。它不做后台冷热迁移。

新增 value store 后要注册工厂，更新 `ResolveEngine` 的合法 value 类型和字段校验。

## IOBackend 接口

`IOBackend` 的单位是 page。`SsdValueStore` 和 SSD index 将嵌套配置转换成后端字段：

| 嵌套字段 | 传给 `IOBackend` 的字段 |
|----------|--------------------------|
| `path` | `file_path` |
| `io.queue_depth` | `queue_cnt` |
| `io.base_offset_bytes / PAGE_SIZE` | `page_id_offset` |
| `io.type` | 工厂类型名 |

新后端至少要实现同步页读写、异步页读写、页对齐 buffer 分配和完成轮询。`BatchReadPages` / `BatchWritePages` 有默认逐页实现；如果后端能批量提交，应覆盖它们。

## 修改检查

改索引：跑 `test_kvengine` 中覆盖该 `index.type` 的用例。

改 value store 或 allocator：跑 `test_kvengine`，必要时补充 `src/test/test_ssd_allocators.cpp`。

改 IO 后端：跑 `test_io_backend`，补跑包含 SSD value 的 `test_kvengine` 场景。

改 `ResolveEngine`：补充拒绝非法字段和非法组合的测试，现有样例在 `src/test/test_kvengine.cpp` 末尾。
