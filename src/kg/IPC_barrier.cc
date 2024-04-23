#include "IPC_barrier.h"

namespace recstore {
void RegisterIPCBarrier(torch::Library &m) {
  m.class_<MultiProcessBarrierHolder>("MultiProcessBarrierHolder")
      .def("Wait", &MultiProcessBarrierHolder::Wait);

  m.class_<MultiProcessBarrierFactory>("MultiProcessBarrierFactory")
      .def_static("Create", &MultiProcessBarrierFactory::CreateStatic)
      .def_static("ClearIPCMemory",
                  &MultiProcessBarrierFactory::ClearIPCMemoryStatic);
}
}  // namespace recstore