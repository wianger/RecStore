#include <folly/executors/CPUThreadPoolExecutor.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/base.h"
#include "base/timer.h"
#include "cache_ps_impl.h"
#include "flatc.h"
#include "parameters.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"

#include "base_ps_server.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using xmhps::CommandRequest;
using xmhps::CommandResponse;
using xmhps::GetParameterRequest;
using xmhps::GetParameterResponse;
using xmhps::PSCommand;
using xmhps::PutParameterRequest;
using xmhps::PutParameterResponse;

class ParameterServiceImpl final : public xmhps::ParameterService::Service {
 public:
  ParameterServiceImpl(CachePS *cache_ps) { cache_ps_ = cache_ps; }

 private:
  Status GetParameter(ServerContext *context,
                      const GetParameterRequest *request,
                      GetParameterResponse *reply) override {
    base::ConstArray<uint64_t> keys_array(request->keys());
    bool isPerf = request->has_perf() && request->perf();
    if (isPerf) {
      xmh::PerfCounter::Record("PS Get Keys", keys_array.Size());
    }
    xmh::Timer timer_ps_get_req("PS GetParameter Req");
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    FB_LOG_EVERY_MS(INFO, 1000)
        << "[PS] Getting " << keys_array.Size() << " keys";

  
    for (auto each : keys_array) {
      ParameterPack parameter_pack;
      cache_ps_->GetParameterRun2Completion(each, parameter_pack, 0);
      compressor.AddItem(parameter_pack, &blocks);
    }

    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    reply->mutable_parameter_value()->swap(blocks[0]);

    if (isPerf) {
      timer_ps_get_req.end();
    } else {
      timer_ps_get_req.destroy();
    }
    return Status::OK;
  }

  Status Command(ServerContext *context, const CommandRequest *request,
                 CommandResponse *reply) override {
    if (request->command() == PSCommand::CLEAR_PS) {
      LOG(WARNING) << "[PS Command] Clear All";
      cache_ps_->Clear();
    } else if (request->command() == PSCommand::RELOAD_PS) {
      LOG(WARNING) << "[PS Command] Reload PS";
      CHECK_NE(request->arg1().size(), 0);
      CHECK_NE(request->arg2().size(), 0);
      CHECK_EQ(request->arg1().size(), 1);
      LOG(WARNING) << "model_config_path = " << request->arg1()[0];
      for (int i = 0; i < request->arg2().size(); i++) {
        LOG(WARNING) << fmt::format("emb_file {}: {}", i, request->arg2()[i]);
      }
      std::vector<std::string> arg1;
      for (auto &each : request->arg1()) {
        arg1.push_back(each);
      }
      std::vector<std::string> arg2;
      for (auto &each : request->arg2()) {
        arg2.push_back(each);
      }

      cache_ps_->Initialize(arg1, arg2);
    } else {
      LOG(FATAL) << "invalid command";
    }
    return Status::OK;
  }

  Status PutParameter(ServerContext *context,
                      const PutParameterRequest *request,
                      PutParameterResponse *reply) override {
    const ParameterCompressReader *reader =
        reinterpret_cast<const ParameterCompressReader *>(
            request->parameter_value().data());
    int size = reader->item_size();
    for (int i = 0; i < size; i++) {
      cache_ps_->PutSingleParameter(reader->item(i), 0);
    }
    return Status::OK;
  }

 private:
  CachePS *cache_ps_;
};

namespace recstore {
class GRPCParameterServer : public BaseParameterServer {
 public:
  GRPCParameterServer() = default;

  // void Init(const nlohmann::json &config) {}

  void Run() {
    std::string server_address("0.0.0.0:15000");
    auto cache_ps = std::make_unique<CachePS>(33762591LL, 128, 1*1024*1024*1024LL, 8, 65536);  // 1GB dict
    ParameterServiceImpl service(cache_ps.get());
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
  }
};
}  // namespace recstore

int main(int argc, char **argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);
  nlohmann::json ex = nlohmann::json::parse(R"(
  {
    "pi": 3.141,
    "happy": true
  }
  )");

  recstore::GRPCParameterServer ps;
  ps.Init(ex);
  ps.Run();
  return 0;
}