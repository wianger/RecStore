#include <pybind11/pybind11.h>

#include "timer.h"

namespace py = pybind11;

using namespace xmh;

PYBIND11_MODULE(timer_module, m) {
  py::class_<PerfCounter>(m, "PerfCounter")
      .def_static("Record", &PerfCounter::Record);

  py::class_<Timer>(m, "Timer")
      .def(py::init<std::string, int>())
      .def("Start", &Timer::start)
      .def("End", &Timer::end)
      .def_static("ManualRecordNs", &Timer::ManualRecordNs);

  py::class_<Reporter>(m, "Reporter")
      .def_static("StartReportThread", &Reporter::StartReportThread);
}