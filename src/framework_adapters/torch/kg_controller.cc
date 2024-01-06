
#include "kg_controller.h"

namespace recstore {

GraphEnv *GraphEnv::instance_;
KGCacheController *KGCacheController::instance_;

void RegisterKGCacheController(torch::Library &m) {
  m.class_<KGCacheController>("KGCacheController")
      .def_static("Init", &KGCacheController::Init)
      .def("RegTensorsPerProcess", &KGCacheController::RegTensorsPerProcess)
      .def("ProcessOneStep", &KGCacheController::ProcessOneStep)
      .def("BlockToStepN", &KGCacheController::BlockToStepN)
      .def("StopThreads", &KGCacheController::StopThreads)
      .def("PrintPq", &KGCacheController::PrintPq);
}

}  // namespace recstore
// 