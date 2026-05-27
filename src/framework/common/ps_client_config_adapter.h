#pragma once

#include <string>

#include "base/json.h"
#include "ps/client_options.h"

namespace recstore {

std::string NormalizePSType(std::string ps_type);
PSClientType ResolveFrameworkPSClientType(const json& config);
json ResolveFrameworkDistributedClientConfig(const json& config);
bool HasFrameworkDistributedClientConfig(const json& config);
json ResolveFrameworkPSClientTransportConfig(const json& config);
PSClientCreateOptions
ResolvePSClientOptionsFromFrameworkConfig(const json& config);

} // namespace recstore
