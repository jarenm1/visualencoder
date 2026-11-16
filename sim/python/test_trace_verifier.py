"""Trace injection edges: asynchronous GPU copies must not evade host scopes."""
import json
import tempfile
import unittest
from pathlib import Path

from sim_rl.profiling import verify_trace


def event(name, category, ts, duration=1, pid=1, tid=1, args=None):
    return {"ph": "X", "name": name, "cat": category, "ts": ts, "dur": duration,
            "pid": pid, "tid": tid, "args": {} if args is None else args}


class TraceVerifierTest(unittest.TestCase):
    def verify(self, additional=()):
        events = [event("sim.collect", "user_annotation", 100, 100),
                  event("cuLaunchKernel", "cuda_driver", 110, args={"correlation": 1}),
                  event("control", "kernel", 220, 7, pid=0, tid=7, args={"correlation": 1})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events + list(additional)}))
            return verify_trace(path)

    def test_native_stages_match_exact_kernel_names(self):
        result = self.verify([
            event("cuLaunchKernel", "cuda_driver", 120, args={"correlation": 2}),
            event("finish", "kernel", 230, 3, pid=0, tid=7, args={"correlation": 2}),
            event("cuLaunchKernel", "cuda_driver", 130, args={"correlation": 3}),
            event("control_flow_helper", "kernel", 240, 11, pid=0, tid=7, args={"correlation": 3})])
        self.assertEqual(result["cuda_stage_microseconds"]["task"], 10)
        self.assertEqual(result["cuda_stage_microseconds"]["other_cuda"], 11)

    def test_generic_memcpy_api_is_classified_by_later_gpu_direction(self):
        for direction in ("Memcpy HtoD (Pageable -> Device)", "Memcpy DtoH (Device -> Pinned)",
                          "Memcpy (Host -> Device)", "Memcpy (Device -> Host)"):
            with self.subTest(direction=direction), self.assertRaisesRegex(RuntimeError, "zero-staging"):
                self.verify([
                    event("cudaMemcpyAsync", "cuda_runtime", 120, args={"correlation": 2}),
                    event(direction, "gpu_memcpy", 300, pid=0, tid=7, args={"correlation": 2})])

    def test_external_id_links_transfer_after_host_collection(self):
        with self.assertRaisesRegex(RuntimeError, "zero-staging"):
            self.verify([
                event("cudaMemcpyAsync", "cuda_runtime", 120, args={"External id": 42}),
                event("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", 300,
                      pid=0, tid=7, args={"External id": 42})])

    def test_host_synchronization_in_collection_is_rejected(self):
        for name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
                     "cuStreamSynchronize", "cuCtxSynchronize"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "zero-staging"):
                self.verify([event(name, "cuda_runtime", 150)])

    def test_d2d_gpu_waits_and_explicit_end_batch_wait_are_allowed(self):
        result = self.verify([
            event("cudaMemcpyAsync", "cuda_runtime", 120, args={"correlation": 2}),
            event("Memcpy DtoD (Device -> Device)", "gpu_memcpy", 300,
                  pid=0, tid=7, args={"correlation": 2}),
            event("cuStreamWaitEvent", "cuda_driver", 130),
            event("cudaStreamSynchronize", "cuda_runtime", 201),
            event("Memcpy HtoD (Pageable -> Device)", "gpu_memcpy", 50,
                  pid=0, tid=7, args={"correlation": 100})])
        self.assertEqual(result["host_transfers_or_waits"], 0)


if __name__ == "__main__":
    unittest.main()
