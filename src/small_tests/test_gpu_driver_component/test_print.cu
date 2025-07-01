#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <thread>

#include "gpreempt.h"

__global__ void print_hello_kernel(int times, int interval_sec) {
  for (int i = 0; i < times; ++i) {
    printf("hello world from kernel, iteration %d\n", i + 1);
    clock_t start_clock = clock();
    clock_t wait_clocks = interval_sec * CLOCKS_PER_SEC;
    while (clock() - start_clock < wait_clocks)
      ;
  }
}

#define util_gettid() ((pid_t)syscall(SYS_gettid))

thread_local int fd = -1;

NV_STATUS NvRmControl(NvHandle hClient, NvHandle hObject, NvU32 cmd,
                      NvP64 params, NvU32 paramsSize) {
  if (fd < 0) {
    fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) {
      return NV_ERR_GENERIC;
    }
  }
  NVOS54_PARAMETERS controlArgs;
  controlArgs.hClient = hClient;
  controlArgs.hObject = hObject;
  controlArgs.cmd = cmd;
  controlArgs.params = params;
  controlArgs.paramsSize = paramsSize;
  controlArgs.flags = 0x0;
  controlArgs.status = 0x0;
  ioctl(fd, OP_CONTROL, &controlArgs);
  return controlArgs.status;
}

NV_STATUS NvRmQuery(NvContext *pContext) {
  if (fd < 0) {
    fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) {
      return NV_ERR_GENERIC;
    }
  }
  NVOS54_PARAMETERS queryArgs;
  queryArgs.hClient = pContext->hClient;
  queryArgs.status = 0x0;
  queryArgs.params = (NvP64)&pContext->channels;
  ioctl(fd, OP_QUERY, &queryArgs);
  pContext->hClient = queryArgs.hClient;
  pContext->hObject = queryArgs.hObject;
  return queryArgs.status;
}

NV_STATUS NvRmDebug(NvContext ctx) {
  NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS suspendParams;
  preemptParams.bWait = NV_FALSE;
  preemptParams.bManualTimeout = NV_FALSE;
  return NvRmControl(ctx.hClient, ctx.hObject, NVA06C_CTRL_CMD_PREEMPT,
                     (NvP64)&suspendParams, sizeof(suspendParams));
}

int main() {
  int times = 100000000;
  int interval_sec = 10;

  print_hello_kernel<<<1, 1>>>(times, interval_sec);

  std::this_thread::sleep_for(std::chrono::seconds(5));

  NvContext nvctx;
  nvctx.hClient = util_gettid();
  NVRMCHECK(NvRmQuery(&nvctx));

  cudaDeviceSynchronize();

  return 0;
}