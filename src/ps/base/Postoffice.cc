#include "Postoffice.h"

#include "base/log.h"

#include <mutex>

DEFINE_int32(num_server_processes, 1, "# of server processes");
DEFINE_int32(num_client_processes, 1, "# of client processes");
DEFINE_int32(global_id, 0, "");

XPostoffice::XPostoffice() {
  num_servers_ = FLAGS_num_server_processes;
  num_clients_ = FLAGS_num_client_processes;

  int g_id = FLAGS_global_id;

  static std::mutex m;
  std::lock_guard<std::mutex> _(m);
  static bool init_ = false;

  CHECK(init_ == false);
  init_ = true;

  global_id_ = g_id;
  if (0 <= g_id && g_id < num_servers_) {
    actor_     = ACTOR_SERVER;
    server_id_ = g_id;
  } else if (num_servers_ <= g_id && g_id < num_servers_ + num_clients_) {
    actor_     = ACTOR_CLIENT;
    client_id_ = g_id - num_servers_;
  } else {
    LOG(FATAL) << "Invalid XPostoffice" << std::endl
               << "global_id = " << global_id_ << std::endl
               << "server_id = " << server_id_ << std::endl
               << "client_id = " << client_id_ << std::endl;
  }
}
