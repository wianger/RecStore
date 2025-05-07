#pragma once
#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "ctrl.h"
#include "buffer.h"
#include "settings.h"
#include "event.h"
#include "queue.h"
#include <cuda_runtime.h>
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif
#include <folly/GLog.h>

#define XMH_CUDA_CHECK(val) \
  { xmh_nv::cuda_check_((val), __FILE__, __LINE__); }

namespace xmh_nv {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string &what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) + ": CUDA error " +
                        std::to_string(val) + ": " + cudaGetErrorString(val));
  }
}
}  // namespace xmh_nv

using error = std::runtime_error;
using std::string;

struct __align__(64) CmdTime {
  size_t size;
  uint64_t submitTime;
  uint64_t completeTime;
  uint64_t moveTime;
};

__host__ static std::shared_ptr<CmdTime> createReportingList(size_t numEntries,
                                                             int device) {
  auto err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    throw err;
  }

  CmdTime *list = nullptr;
  err = cudaMalloc(&list, sizeof(CmdTime) * numEntries);
  if (err != cudaSuccess) {
    throw err;
  }

  return std::shared_ptr<CmdTime>(list, cudaFree);
}

__host__ static std::shared_ptr<CmdTime> createReportingList(size_t numEntries) {
  CmdTime *list = nullptr;

  auto err = cudaHostAlloc(&list, sizeof(CmdTime) * numEntries, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw err;
  }

  return std::shared_ptr<CmdTime>(list, cudaFreeHost);
}

__device__ static void moveBytes(const void *src, size_t srcOffset, void *dst,
                                 size_t dstOffset, size_t size) {
  const uint16_t numThreads = blockDim.x;
  const uint16_t threadNum = threadIdx.x;

  const ulong4 *source = (ulong4 *)(((const unsigned char *)src) + srcOffset);
  ulong4 *destination = (ulong4 *)(((unsigned char *)dst) + dstOffset);

  for (size_t i = 0, n = size / sizeof(ulong4); i < n; i += numThreads) {
    destination[i + threadNum] = source[i + threadNum];
  }
}

__device__ static void waitForIoCompletion(nvm_queue_t *cq, nvm_queue_t *sq,
                                           int *errCode) {
  const uint16_t numThreads = blockDim.x;

  for (uint16_t i = 0; i < numThreads; ++i) {
    nvm_cpl_t *cpl = nullptr;
    while ((cpl = nvm_cq_dequeue(cq)) == nullptr)
      ;

    nvm_sq_update(sq);

    if (!NVM_ERR_OK(cpl)) {
      //*errCount = *errCount + 1;
      *errCode = NVM_ERR_PACK(cpl, 0);
    }
  }

  nvm_cq_update(cq);
}

__device__ static nvm_cmd_t *prepareChunk(QueuePair *qp, nvm_cmd_t *last,
                                          const uint64_t ioaddr, uint16_t offset,
                                          uint64_t blockOffset, uint32_t currChunk) {
  nvm_cmd_t local;
  const uint16_t numThreads = blockDim.x;
  const uint16_t threadNum = threadIdx.x;
  const uint16_t threadOffset = threadNum + numThreads * offset;

  const uint32_t pageSize = qp->pageSize;
  const uint32_t blockSize = qp->blockSize;
  const uint32_t nvmNamespace = qp->nvmNamespace;
  const uint32_t chunkPages = qp->pagesPerChunk;

  // Calculate offsets
  const uint16_t blocksPerChunk = NVM_PAGE_TO_BLOCK(pageSize, blockSize, chunkPages);
  const uint64_t currBlock =
      NVM_PAGE_TO_BLOCK(pageSize, blockSize, (currChunk + threadNum) * chunkPages);

  // Prepare PRP list building
  void *prpListPtr = NVM_PTR_OFFSET(qp->prpList, pageSize, threadOffset);
  uint64_t prpListAddr = NVM_ADDR_OFFSET(qp->prpListIoAddr, pageSize, threadOffset);
  nvm_prp_list_t prpList = NVM_PRP_LIST_INIT(prpListPtr, true, pageSize, prpListAddr);

  uint64_t addrs[0x1000 / sizeof(uint64_t)];  // FIXME: This assumes that page size is 4K
  for (uint32_t page = 0; page < chunkPages; ++page) {
    addrs[page] = NVM_ADDR_OFFSET(ioaddr, pageSize, chunkPages * threadOffset + page);
  }

  // Enqueue commands
  nvm_cmd_t *cmd = nvm_sq_enqueue_n(&qp->sq, last, numThreads, threadNum);

  // Set command fields
  nvm_cmd_header(&local, threadNum, NVM_IO_READ, nvmNamespace);
  nvm_cmd_data(&local, 1, &prpList, chunkPages, addrs);
  nvm_cmd_rw_blks(&local, currBlock + blockOffset, blocksPerChunk);

  *cmd = local;
  __threadfence();
  return cmd;
}

__global__ static void moveKernel(void *src, void *dst, size_t chunkSize) {
  const uint16_t numThreads = blockDim.x;
  moveBytes(src, 0, dst, 0, chunkSize * numThreads);
}

__host__ static inline void launchMoveKernel(size_t pageSize, void *input, void *src,
                                             void *dst, size_t currChunk,
                                             const Settings &settings) {
  const auto numPages = settings.numPages;
  const auto numThreads = settings.numThreads;
  const auto chunkSize = pageSize * numPages;

  void *dstPtr = (void *)(((unsigned char *)dst) + chunkSize * currChunk);
  void *inputPtr = (void *)(((unsigned char *)input) + chunkSize * currChunk);

  cudaError_t err =
      cudaMemcpyAsync(src, inputPtr, chunkSize * numThreads, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw err;
  }

  moveKernel<<<1, numThreads>>>(src, dstPtr, chunkSize);
}

static double LoopH2DCopyAndMoveKernel(void *fileMap, BufferPtr destination,
                                       size_t pageSize, const Settings &settings) {
  const size_t chunkSize = pageSize * settings.numPages;
  const size_t numThreads = settings.numThreads;
  const size_t totalChunks = settings.numChunks * numThreads;

  const size_t sourceBufferSize = NVM_PAGE_ALIGN(chunkSize * numThreads, 1UL << 16);
  auto source = createBuffer(sourceBufferSize, settings.cudaDevice);

  auto err = cudaSetDevice(settings.cudaDevice);
  if (err != cudaSuccess) {
    throw err;
  }

  Event before;
  for (size_t currChunk = 0; currChunk < totalChunks; currChunk += numThreads) {
    launchMoveKernel(pageSize, fileMap, source.get(), destination.get(), currChunk,
                     settings);
  }
  Event after;

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw err;
  }

  return after - before;
}

__global__ static void readDoubleBuffered(QueuePair *qp, const uint64_t ioaddr, void *src,
                                          void *dst, size_t numChunks,
                                          uint64_t startBlock, int *errCode,
                                          CmdTime *times) {
  const uint16_t numThreads = blockDim.x;
  const uint16_t threadNum = threadIdx.x;
  const uint32_t pageSize = qp->pageSize;
  const size_t chunkSize = qp->pagesPerChunk * pageSize;
  nvm_queue_t *sq = &qp->sq;

  uint64_t blockOffset = startBlock;

  uint32_t currChunk = 0;
  bool bufferOffset = false;
  uint32_t i = 0;

  nvm_cmd_t *last =
      prepareChunk(qp, nullptr, ioaddr, bufferOffset, blockOffset, currChunk);

  auto beforeSubmit = clock();
  if (threadNum == 0) {
    *errCode = 0;
    nvm_sq_submit(sq);
  }
  __syncthreads();

  while (currChunk + numThreads < numChunks) {
    // Prepare in advance next chunk
    last = prepareChunk(qp, last, ioaddr, !bufferOffset, blockOffset,
                        currChunk + numThreads);

    // Consume completions for the previous window
    beforeSubmit = clock();
    if (threadNum == 0) {
      waitForIoCompletion(&qp->cq, sq, errCode);
      nvm_sq_submit(sq);
    }
    __syncthreads();
    auto afterSync = clock();

    // Move received chunk
    moveBytes(src, bufferOffset * numThreads * chunkSize, dst, currChunk * chunkSize,
              chunkSize * numThreads);
    auto afterMove = clock();

    // Record statistics
    if (times != nullptr && threadNum == 0) {
      CmdTime *t = &times[i];
      t->size = chunkSize * numThreads;
      t->submitTime = beforeSubmit;
      t->completeTime = afterSync;
      t->moveTime = afterMove;
    }
    __syncthreads();

    // Update position and input buffer
    bufferOffset = !bufferOffset;
    currChunk += numThreads;
    ++i;
  }

  // Wait for final buffer to complete
  if (threadNum == 0) {
    waitForIoCompletion(&qp->cq, sq, errCode);
  }
  __syncthreads();
  auto afterSync = clock();

  moveBytes(src, bufferOffset * numThreads * chunkSize, dst, currChunk * chunkSize,
            chunkSize * numThreads);
  auto afterMove = clock();

  // Record statistics
  if (times != nullptr && threadNum == 0) {
    CmdTime *t = &times[i];
    t->size = chunkSize * numThreads;
    t->submitTime = beforeSubmit;
    t->completeTime = afterSync;
    t->moveTime = afterMove;
  }
}

__global__ static void readSingleBuffered(QueuePair *qp, const uint64_t ioaddr, void *src,
                                          void *dst, size_t numChunks,
                                          uint64_t startBlock, int *errCode,
                                          CmdTime *times) {
  const uint16_t numThreads = blockDim.x;
  const uint16_t threadNum = threadIdx.x;
  const uint32_t pageSize = qp->pageSize;
  const size_t chunkSize = qp->pagesPerChunk * pageSize;
  nvm_queue_t *sq = &qp->sq;

  uint64_t blockOffset = startBlock;

  uint32_t currChunk = 0;
  uint32_t i = 0;

  nvm_cmd_t *cmd = nullptr;

  if (threadNum == 0) {
    *errCode = 0;
  }
  __syncthreads();

  while (currChunk < numChunks) {
    // Prepare in advance next chunk
    cmd = prepareChunk(qp, cmd, ioaddr, 0, blockOffset, currChunk);

    // Consume completions for the previous window
    auto beforeSubmit = clock();
    if (threadNum == 0) {
      nvm_sq_submit(sq);
      waitForIoCompletion(&qp->cq, sq, errCode);
    }
    __syncthreads();
    auto afterSync = clock();

    // Move received chunk
    moveBytes(src, 0, dst, currChunk * chunkSize, chunkSize * numThreads);
    auto afterMove = clock();

    // Record statistics
    if (times != nullptr && threadNum == 0) {
      CmdTime *t = &times[i];
      t->size = chunkSize * numThreads;
      t->submitTime = beforeSubmit;
      t->completeTime = afterSync;
      t->moveTime = afterMove;
    }
    __syncthreads();

    // Update position and input buffer
    currChunk += numThreads;
    ++i;
  }
}

static void printStatistics(const Settings &settings, const cudaDeviceProp &prop,
                            const std::shared_ptr<CmdTime> gpuTimes) {
  const size_t numChunks = settings.numChunks;
  auto hostTimes = createReportingList(numChunks);

  auto err = cudaMemcpy(hostTimes.get(), gpuTimes.get(), sizeof(CmdTime) * numChunks,
                        cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw err;
  }

  const auto *times = hostTimes.get();
  const double rate = ((double)prop.clockRate) / 1e3;  // GPU's clock frequency in MHz

  fprintf(stdout, "#%9s; %12s; %12s; %12s; %12s; %12s; %12s;\n", "size", "disk_lat",
          "disk_bw", "mem_lat", "mem_bw", "cum_lat", "cum_bw");
  fflush(stdout);
  for (size_t i = 0; i < numChunks; ++i) {
    const auto &t = times[i];
    auto diskTime = (t.completeTime - t.submitTime) / rate;
    auto moveTime = (t.moveTime - t.completeTime) / rate;
    auto totalTime = (t.moveTime - t.submitTime) / rate;

    auto diskBw = times[i].size / diskTime;
    auto moveBw = times[i].size / moveTime;
    auto totalBw = times[i].size / totalTime;

    fprintf(stdout, "%10zu; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f;\n", t.size,
            diskTime, diskBw, moveTime, moveBw, totalTime, totalBw);
    fflush(stdout);
  }
}

static void outputFile(BufferPtr data, size_t size, const char *filename) {
  auto buffer = createBuffer(size);

  cudaError_t err = cudaMemcpy(buffer.get(), data.get(), size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw error(string("Failed to copy data from destination: ") +
                cudaGetErrorString(err));
  }

  FILE *fp = fopen(filename, "wb");
  fwrite(buffer.get(), 1, size, fp);
  fclose(fp);
}

static int useBlockDevice(const Settings &settings, const cudaDeviceProp &properties) {
  int fd = open(settings.blockDevicePath, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Failed to open block device: %s\n", strerror(errno));
    return 1;
  }

  const size_t pageSize = sysconf(_SC_PAGESIZE);
  const size_t blockSize = 512;  // FIXME: specify this from command line
  const size_t totalChunks = settings.numChunks * settings.numThreads;
  const size_t totalPages = totalChunks * settings.numPages;

  fprintf(stderr, "CUDA device           : %u %s (%s)\n", settings.cudaDevice,
          properties.name, settings.getDeviceBDF().c_str());
#ifdef __DIS_CLUSTER__
  fprintf(stderr, "CUDA device fdid      : %lx\n", settings.cudaDeviceId);
#endif
  fprintf(stderr, "Controller page size  : %zu B\n", pageSize);
  fprintf(stderr, "Assumed block size    : %zu B\n", blockSize);
  fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
  fprintf(stderr, "Chunks per thread     : %zu\n", settings.numChunks);
  fprintf(stderr, "Pages per chunk       : %zu\n", settings.numPages);
  fprintf(stderr, "Total number of pages : %zu\n", totalPages);
  fprintf(stderr, "Double buffering      : %s\n", settings.doubleBuffered ? "yes" : "no");

  void *ptr = mmap(nullptr, totalPages * pageSize, PROT_READ, MAP_FILE | MAP_PRIVATE, fd,
                   settings.startBlock * blockSize);
  if (ptr == nullptr || ptr == MAP_FAILED) {
    close(fd);
    fprintf(stderr, "Failed to memory map block device: %s\n", strerror(errno));
    return 1;
  }

  try {
    auto outputBuffer = createBuffer(totalPages * pageSize);

    double usecs = LoopH2DCopyAndMoveKernel(ptr, outputBuffer, pageSize, settings);

    fprintf(stderr, "Event time elapsed    : %.3f µs\n", usecs);
    fprintf(stderr, "Estimated bandwidth   : %.3f MiB/s\n",
            (totalPages * pageSize) / usecs);

    if (settings.output != nullptr) {
      outputFile(outputBuffer, totalPages * pageSize, settings.output);
    }
  } catch (const cudaError_t err) {
    munmap(ptr, totalPages * pageSize);
    close(fd);
    fprintf(stderr, "Unexpected CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  } catch (const error &e) {
    munmap(ptr, totalPages * pageSize);
    close(fd);
    fprintf(stderr, "Unexpected error: %s\n", e.what());
    return 1;
  }

  munmap(ptr, totalPages * pageSize);
  close(fd);
  return 0;
}

class BaMWrapper {
  const std::string controllerPath = "/dev/libnvm0";

 public:
  BaMWrapper() : ctrl(controllerPath.c_str(), 1) {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess) {
      LOG(FATAL) << "Failed to get CUDA device properties";
    }
    // settings
    settings.controllerPath = controllerPath.c_str();
    settings.doubleBuffered = true;
    // settings done

    if (settings.blockDevicePath != nullptr) {
      useBlockDevice(settings, properties);
    }

    verifyCudaDevice(settings.cudaDevice);
    verifyNumberOfThreads(settings.numThreads);

    settings.domain = properties.pciDomainID;
    settings.bus = properties.pciBusID;
    settings.devfn = properties.pciDeviceID;

    ctrl.reserveQueues(1);

    const size_t pageSize = ctrl.info.page_size;
    const size_t blockSize = ctrl.ns.lba_data_size;
    const size_t chunkSize = pageSize * settings.numPages;
    const size_t totalChunks = settings.numChunks * settings.numThreads;
    const size_t totalPages = totalChunks * settings.numPages;
    const size_t totalBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, totalPages);

    CHECK_LE(chunkSize, ctrl.info.max_data_size)
        << "Chunk size can not be larger than controller data size";
    CHECK_LE(totalBlocks, ctrl.ns.size) << "Requesting read size larger than disk size";

    fprintf(stderr, "CUDA device           : %u %s (%s)\n", settings.cudaDevice,
            properties.name, settings.getDeviceBDF().c_str());
    fprintf(stderr, "Controller page size  : %zu B\n", pageSize);
    fprintf(stderr, "Namespace block size  : %zu B\n", blockSize);
    fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
    fprintf(stderr, "Chunks per thread     : %zu\n", settings.numChunks);
    fprintf(stderr, "Pages per chunk       : %zu\n", settings.numPages);
    fprintf(stderr, "Total number of pages : %zu\n", totalPages);
    fprintf(stderr, "Total number of blocks: %zu\n", totalBlocks);
    fprintf(stderr, "Double buffering      : %s\n",
            settings.doubleBuffered ? "yes" : "no");

    auto outputBuffer =
        createBuffer(ctrl.info.page_size * totalPages, settings.cudaDevice);

    cudaError_t err = cudaHostRegister((void *)ctrl.ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE,
                                       cudaHostRegisterIoMemory);

    if (err != cudaSuccess) {
      LOG(FATAL) << "Unexpected error while mapping IO memory (cudaHostRegister): "
                 << cudaGetErrorString(err);
    }
    XMH_CUDA_CHECK(cudaMalloc(&d_errorCode, sizeof(int)));
    if (settings.stats) {
      times = createReportingList(settings.numChunks, settings.cudaDevice);
    }
  }

  void InitQueuePair() {
    QueuePair queuePair;
    DmaPtr queueMemory = prepareQueuePair(queuePair, ctrl, settings);
    // Set up and prepare queues
    deviceQueue = createBuffer(sizeof(QueuePair), settings.cudaDevice);
    XMH_CUDA_CHECK(cudaMemcpy(deviceQueue.get(), &queuePair, sizeof(QueuePair),
                              cudaMemcpyHostToDevice));
    XMH_CUDA_CHECK(cudaSetDevice(settings.cudaDevice));
  }

  void CheckError() const {
    int h_errorCode = 0;
    XMH_CUDA_CHECK(
        cudaMemcpy(&h_errorCode, d_errorCode, sizeof(int), cudaMemcpyDeviceToHost));
    XMH_CUDA_CHECK(cudaFree(d_errorCode));

    CHECK_EQ(h_errorCode, 0) << "WARNING: There were NVM errors: "
                             << nvm_strerror(h_errorCode);
  }

  void Write(void *h_input, uint64_t startBlock, int block_num){

  }


  // d_d_outputBuffer's size is <ctrl.info.page_size * totalPages>
  double Query(void *d_outputBuffer, uint64_t startBlock, int block_num) {
    const size_t pageSize = ctrl.info.page_size;
    const size_t chunkSize = pageSize * settings.numPages;
    const size_t totalChunks = settings.numChunks * settings.numThreads;

    // Create input buffer
    const size_t sourceBufferSize = NVM_PAGE_ALIGN(
        (settings.doubleBuffered + 1) * chunkSize * settings.numThreads, 1UL << 16);
    // NOTE(xieminhui): adapter & segmentID can be ignored
    auto source =
        createDma(ctrl.ctrl, sourceBufferSize, settings.cudaDevice, settings.adapter,
                  settings.segmentId + 1);  // vaddr is a dev ptr

    // Launch kernel
    Event before;
    if (settings.doubleBuffered) {
      readDoubleBuffered<<<1, settings.numThreads>>>(
          (QueuePair *)deviceQueue.get(), source->ioaddrs[0], source->vaddr,
          d_outputBuffer, totalChunks, startBlock, d_errorCode, times.get());
    } else {
      readSingleBuffered<<<1, settings.numThreads>>>(
          (QueuePair *)deviceQueue.get(), source->ioaddrs[0], source->vaddr,
          d_outputBuffer, totalChunks, startBlock, d_errorCode, times.get());
    }
    Event after;
    XMH_CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = after - before;

    CheckError();
    if (settings.stats) {
      cudaDeviceProp properties;
      XMH_CUDA_CHECK(cudaGetDeviceProperties(&properties, settings.cudaDevice));
      printStatistics(settings, properties, times);
    }
    return elapsed;
  }

  static void outputFile(void *d_data, size_t size, const char *filename) {
    auto buffer = createBuffer(size);

    cudaError_t err = cudaMemcpy(buffer.get(), d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw error(string("Failed to copy data from destination: ") +
                  cudaGetErrorString(err));
    }

    FILE *fp = fopen(filename, "wb");
    fwrite(buffer.get(), 1, size, fp);
    fclose(fp);
  }

 private:
  int *d_errorCode = nullptr;
  std::shared_ptr<CmdTime> times;

  static void verifyNumberOfThreads(size_t numThreads) {
    size_t i = 0;
    while ((1ULL << i) <= 32) {
      if ((1ULL << i) == numThreads) {
        return;
      }
      ++i;
    }
    LOG(FATAL) << "Invalid number of threads, must be a power of 2";
  }

  static void verifyCudaDevice(int device) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
      LOG(FATAL) << "Unexpected error: " << cudaGetErrorString(err);
    }
    if (device < 0 || device >= deviceCount) {
      LOG(FATAL) << "Invalid CUDA device: " << std::to_string(device);
    }
  }

  Settings settings;
  Controller ctrl;
  BufferPtr deviceQueue;
};