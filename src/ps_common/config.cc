#include "config.h"

DEFINE_int32(value_size, 32 * 4, "");
DEFINE_int32(max_kv_num_per_request, 300,
             "max kv_ count per request, used for allocate buffer");
