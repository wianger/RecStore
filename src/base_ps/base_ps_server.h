#include "base/array.h"
#include "base/base.h"
#include "base/json.h"
#include "base/timer.h"

namespace recstore {
class BaseParameterServer {
 public:
  virtual void Init(const json& config) { config_ = config; }
  virtual void Run() = 0;

 protected:
  json config_;
};
}  // namespace recstore