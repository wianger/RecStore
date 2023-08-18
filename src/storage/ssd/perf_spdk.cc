#include <folly/Random.h>
#include <folly/init/Init.h>

#include <atomic>
#include <iostream>
#include <vector>

#include "base/timer.h"
#include "spdk_wrapper.h"

DEFINE_int32(query_count, 100, "# of query embs in one round");
DEFINE_int32(key_space_M, 1, "key space in millions");

void cb(void *ctx, const struct spdk_nvme_cpl *cpl) {
  std::atomic<int> *p = (std::atomic<int> *)ctx;
  p->fetch_add(1);
}

int main(int argc, char **argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread();
  auto ssd = ssdps::SpdkWrapper::create(1);
  ssd->Init();

  int batch_get_num = FLAGS_query_count;
  int lba_size = ssd->GetLBASize();

  char *buf = (char *)spdk_zmalloc(lba_size * batch_get_num, 0, NULL,
                                   SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);

  std::vector<int> batch_get_lba_id(batch_get_num);

  while (1) {
    for (int i = 0; i < batch_get_num; i++) {
      batch_get_lba_id[i] = folly::Random::rand32(FLAGS_key_space_M * 1e6);
    }
    xmh::Timer timer("get");
    std::atomic<int> counter{0};
    for (int i = 0; i < batch_get_num; i++) {
      ssd->SubmitReadCommand(buf + lba_size * i, lba_size, batch_get_lba_id[i],
                             cb, &counter, 0);
    }
    while (counter != batch_get_num) ssd->PollCompleteQueue(0);
    timer.end();
  }

  return 0;
}
