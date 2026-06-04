import importlib.util
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("analyze_embupdate_stages.py")
    spec = importlib.util.spec_from_file_location("analyze_embupdate_stages", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyzeEmbupdateStagesTest(unittest.TestCase):
    def test_build_merged_request_map_falls_back_to_ordered_prefix_pairing(self) -> None:
        mod = _load_module()
        by_trace = {
            "op_client::EmbUpdate|100": {
                "op_total_us": 50_000.0,
                "op_validate_us": 2.0,
            },
            "brpc_client::EmbUpdate|100": {
                "client_total_us": 40_000.0,
                "client_serialize_us": 2_000.0,
                "client_rpc_us": 30_000.0,
            },
            "brpc_server::EmbUpdate|120": {
                "server_total_us": 8_000.0,
                "server_backend_update_us": 6_500.0,
            },
            "op_client::EmbUpdate|200": {
                "op_total_us": 55_000.0,
                "op_validate_us": 2.0,
            },
            "brpc_client::EmbUpdate|200": {
                "client_total_us": 42_000.0,
                "client_serialize_us": 2_100.0,
                "client_rpc_us": 31_000.0,
            },
            "brpc_server::EmbUpdate|220": {
                "server_total_us": 9_000.0,
                "server_backend_update_us": 7_200.0,
            },
        }

        merged = mod.build_merged_request_map(by_trace)

        self.assertEqual(sorted(merged.keys()), ["100", "200"])
        self.assertEqual(merged["100"]["server_total_us"], 8_000.0)
        self.assertEqual(merged["200"]["server_backend_update_us"], 7_200.0)

        derived_100 = mod.derive_chain_metrics(merged["100"])
        self.assertEqual(derived_100["network_transport_us"], 22_000.0)
        self.assertEqual(derived_100["server_framework_overhead_us"], 1_500.0)

    def test_build_trace_map_accepts_custom_table_name(self) -> None:
        mod = _load_module()
        events = [
            {
                "table_name": "embupdate_stages",
                "unique_id": "op_client::EmbUpdate|1",
                "metric_name": "op_total_us",
                "metric_value": 100.0,
            },
            {
                "table_name": "local_shm_server_stages",
                "unique_id": "local_shm_req|2",
                "metric_name": "server_process_total_us",
                "metric_value": 200.0,
            },
        ]

        by_trace = mod.build_trace_map(events, "", "local_shm_server_stages")

        self.assertEqual(list(by_trace.keys()), ["local_shm_req|2"])
        self.assertEqual(by_trace["local_shm_req|2"]["server_process_total_us"], 200.0)


if __name__ == "__main__":
    unittest.main()
