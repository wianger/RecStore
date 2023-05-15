
#include <folly/Format.h>
#include <folly/GLog.h>
#include <folly/Likely.h>

#include "spdk_wrapper.h"

namespace ssdps {

class SpdkWrapperImplementation : public SpdkWrapper {
 public:
  void Init() override {
    LOG(INFO) << "Initializing NVMe Controllers";
    spdk_env_opts_init(&opts_);

    CHECK(spdk_env_init(&opts_) >= 0) << "Unable to initialize SPDK env\n";

    spdk_nvme_trid_populate_transport(&g_trid_, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(g_trid_.subnqn, sizeof(g_trid_.subnqn), "%s",
             SPDK_NVMF_DISCOVERY_NQN);

    CHECK_EQ(spdk_nvme_probe(
                 &g_trid_, this, SpdkWrapperImplementation::ProbeCallBack,
                 SpdkWrapperImplementation::AttachCallBack, nullptr),
             0)
        << "spdk_nvme_probe failed";

    CHECK_NE(g_controllers_.size(), 0) << "no NVMe controllers found";
    LOG(INFO) << "Initialization complete";

    for (auto &ns_entry : g_namespaces_) {
      ns_entry.qpair = spdk_nvme_ctrlr_alloc_io_qpair(ns_entry.ctrlr, NULL, 0);

      CHECK_NE(ns_entry.qpair, nullptr)
          << "ERROR: spdk_nvme_ctrlr_alloc_io_qpair() failed";
    }
    CHECK_EQ(g_namespaces_.size(), 1) << "KISS, now only support 1 namespace";
  }

  ~SpdkWrapperImplementation() {
    for (auto ns_entry : g_namespaces_) {
      spdk_nvme_ctrlr_free_io_qpair(ns_entry.qpair);
      struct spdk_nvme_detach_ctx *detach_ctx = NULL;
      spdk_nvme_detach_async(ns_entry.ctrlr, &detach_ctx);
      if (detach_ctx) {
        spdk_nvme_detach_poll(detach_ctx);
      }
    }
  }

  int SubmitWriteCommand(const void *pinned_src, const int64_t bytes,
                         const int64_t lba, spdk_nvme_cmd_cb func,
                         void *ctx) override {
    auto ns_entry = g_namespaces_[0];
    uint32_t lba_count = (bytes + kLBASize_ - 1) / kLBASize_;

    int ret =
        spdk_nvme_ns_cmd_write(ns_entry.ns, ns_entry.qpair, (void *)pinned_src,
                               lba, lba_count, func, ctx, 0);

    CHECK(ret == 0 || ret == -ENOMEM) << ret;
    return ret;
  }

  void SubmitReadCommand(void *pinned_dst, const int64_t bytes,
                         const int64_t lba, spdk_nvme_cmd_cb func,
                         void *ctx) override {
    auto ns_entry = g_namespaces_[0];
    uint32_t lba_count = (bytes + kLBASize_ - 1) / kLBASize_;
    while (1) {
      auto ret = spdk_nvme_ns_cmd_read(ns_entry.ns, ns_entry.qpair, pinned_dst,
                                       lba, lba_count, func, ctx, 0);
      if (ret == 0) {
        return;
      } else if (ret == -ENOMEM) {
        FB_LOG_EVERY_MS(ERROR, 10000)
            << "SubmitReadCommand return with ENOMEM, let's poll CQ";
        PollCompleteQueue();
      } else {
        LOG(FATAL) << "SubmitReadCommand Error " << ret;
      }
    }
  }

  FOLLY_ALWAYS_INLINE void PollCompleteQueue() override {
    auto ns_entry = g_namespaces_[0];
    spdk_nvme_qpair_process_completions(ns_entry.qpair, 0);
  }

  FOLLY_ALWAYS_INLINE int GetLBASize() const override { return kLBASize_; }

  FOLLY_ALWAYS_INLINE uint64_t GetLBANumber() const override {
    // uint64_t capacity = 3200631791616LL;
    uint64_t capacity = 300 * 1024 * 1024 * 1024LL;
    CHECK_EQ(0, capacity % GetLBASize());
    return capacity / GetLBASize();
  }

  void SyncRead(void *pinned_dst, const int64_t bytes, const int64_t lba) {
    std::atomic_bool flag{false};
    SubmitReadCommand(pinned_dst, bytes, lba, SyncCommandCompleteCB,
                      (void *)&flag);
    while (!flag.load()) {
      PollCompleteQueue();
    }
  }

  void SyncWrite(const void *pinned_src, const int64_t bytes,
                 const int64_t lba) {
    std::atomic_bool flag{false};
    SubmitWriteCommand(pinned_src, bytes, lba, SyncCommandCompleteCB,
                       (void *)&flag);
    while (!flag.load()) {
      PollCompleteQueue();
    }
  }

  void Sync2Read(void *pinned_dst, const int64_t lba) {
    LOG(FATAL)
        << "dont use this function, this function is only used for test nvme "
           "command fusion";
    std::atomic_bool flag{false};
    auto ns_entry = g_namespaces_[0];

    auto ret =
        spdk_nvme_ns_cmd_read(ns_entry.ns, ns_entry.qpair, pinned_dst, lba, 1,
                              nullptr, nullptr, SPDK_NVME_CMD_FUSE_FIRST);
    if (ret == 0) {
      LOG(INFO) << "submit successful 1";
    } else if (ret == -ENOMEM) {
      FB_LOG_EVERY_MS(ERROR, 10000)
          << "SubmitReadCommand return with ENOMEM, let's poll CQ";
      PollCompleteQueue();
    } else {
      LOG(FATAL) << "SubmitReadCommand Error " << ret;
    }

    ret = spdk_nvme_ns_cmd_read(ns_entry.ns, ns_entry.qpair,
                                (char *)pinned_dst + 512, lba + 1, 1, nullptr,
                                nullptr, SPDK_NVME_CMD_FUSE_SECOND);
    if (ret == 0) {
      LOG(INFO) << "submit successful 2";
    } else if (ret == -ENOMEM) {
      FB_LOG_EVERY_MS(ERROR, 10000)
          << "SubmitReadCommand return with ENOMEM, let's poll CQ";
      PollCompleteQueue();
    } else {
      LOG(FATAL) << "SubmitReadCommand Error " << ret;
    }
  }

 private:
  static void SyncCommandCompleteCB(void *ctx,
                                    const struct spdk_nvme_cpl *cpl) {
    if (FOLLY_UNLIKELY(spdk_nvme_cpl_is_error(cpl))) {
      LOG(FATAL) << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }
    std::atomic_bool *p = (std::atomic_bool *)ctx;
    *p = true;
  }

  // static void CmdCallBack(void *ctx, const struct spdk_nvme_cpl *cpl) {
  //   if (spdk_nvme_cpl_is_error(cpl)) {
  //     LOG(FATAL) << "I/O error status: " <<
  //     spdk_nvme_cpl_get_status_string(&cpl->status);
  //   }
  //   Func f = (Func)ctx;
  //   f();
  // }

  static bool ProbeCallBack(void *cb_ctx,
                            const struct spdk_nvme_transport_id *trid,
                            struct spdk_nvme_ctrlr_opts *opts) {
    LOG(INFO) << "Attaching to " << trid->traddr;
    return true;
  }

  static void AttachCallBack(void *cb_ctx,
                             const struct spdk_nvme_transport_id *trid,
                             struct spdk_nvme_ctrlr *ctrlr,
                             const struct spdk_nvme_ctrlr_opts *opts) {
    SpdkWrapperImplementation *ptr = (SpdkWrapperImplementation *)(cb_ctx);
    LOG(INFO) << folly::format("Attached to {}", trid->traddr);

    /*
     * spdk_nvme_ctrlr is the logical abstraction in SPDK for an NVMe
     *  controller.  During initialization, the IDENTIFY data for the
     *  controller is read using an NVMe admin command, and that data
     *  can be retrieved using spdk_nvme_ctrlr_get_data() to get
     *  detailed information on the controller.  Refer to the NVMe
     *  specification for more details on IDENTIFY for NVMe controllers.
     */
    const struct spdk_nvme_ctrlr_data *cdata = spdk_nvme_ctrlr_get_data(ctrlr);

    std::unique_ptr<char[]> buf(new char[1024]);
    std::snprintf(buf.get(), 1024, "%-20.20s (%-20.20s)", cdata->mn, cdata->sn);
    std::string name(buf.get(), buf.get() + 1024 - 1);

    ptr->g_controllers_[name] = ctrlr;

    /*
     * Each controller has one or more namespaces.  An NVMe namespace is
     * basically equivalent to a SCSI LUN.  The controller's IDENTIFY data tells
     * us how many namespaces exist on the controller.  For Intel(R) P3X00
     * controllers, it will just be one namespace.
     *
     * Note that in NVMe, namespace IDs start at 1, not 0.
     */
    for (int nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
      struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
      if (ns == NULL || !spdk_nvme_ns_is_active(ns)) {
        continue;
      }

      ns_entry entry;
      entry.ctrlr = ctrlr;
      entry.ns = ns;

      ptr->g_namespaces_.push_back(entry);

      LOG(INFO) << folly::format("Namespace ID: {} size: {} GB\n",
                                 spdk_nvme_ns_get_id(ns),
                                 spdk_nvme_ns_get_size(ns) / 1000000000);
    }
  }
  spdk_env_opts opts_;
  spdk_nvme_transport_id g_trid_;
  const int kLBASize_ = 512;
  std::unordered_map<std::string, spdk_nvme_ctrlr *> g_controllers_;

  struct ns_entry {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
  };

  std::vector<ns_entry> g_namespaces_;
};

std::unique_ptr<SpdkWrapper> SpdkWrapper::create() {
  std::unique_ptr<SpdkWrapper> ret =
      std::make_unique<SpdkWrapperImplementation>();
  return ret;
}

}  // namespace ssdps