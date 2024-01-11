#include <folly/String.h>

#include "pprint.h"

namespace base {
std::string PrettyPrintBytes(double bytes) {
  return folly::prettyPrint(bytes, folly::PRETTY_BYTES);
}
}  // namespace base