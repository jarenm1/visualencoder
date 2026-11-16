"""Chrome/Kineto trace verification using CPU scopes and GPU correlations."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch


def _correlation(event):
    args = event.get("args", {})
    for key in ("correlation", "Correlation ID", "correlation id"):
        if key in args:
            return str(args[key])
    return None


def _external(event):
    args = event.get("args", {})
    for key in ("External id", "External Id", "external id"):
        if key in args:
            return str(args[key])
    return None


def _host_transfer(event):
    name = event.get("name", "")
    args = event.get("args", {})
    description = " ".join((name, json.dumps(args))).lower()
    compact = re.sub(r"[^a-z0-9]", "", description)
    host_to_device = (any(marker in compact for marker in ("htod", "h2d", "hosttodevice", "hosttodev"))
                      or re.search(r"host\s*(?:->|to)\s*device", description))
    device_to_host = (any(marker in compact for marker in ("dtoh", "d2h", "devicetohost", "devtohost"))
                      or re.search(r"device\s*(?:->|to)\s*host", description))
    return bool((host_to_device or device_to_host) and
                ("memcpy" in description or "copy" in description or "memory" in description))


def _host_wait(event):
    name = event.get("name", "")
    return bool(re.search(r"(?:cuda|cu)(?:Stream|Device|Event|Ctx)Synchronize", name, re.I))


def verify_trace(trace_path, scope="sim.collect"):
    """GPU activities may execute after the host scope; use launch correlation.

    Temporal overlap is only a fallback for GPU events without correlation IDs.
    A trace missing CUDA runtime or GPU activities is rejected, not certified.
    Device-to-device copies and cuStreamWaitEvent GPU ordering are permitted.
    """
    trace = json.loads(Path(trace_path).read_text())
    events = [event for event in trace.get("traceEvents", []) if event.get("ph") == "X"]
    scopes = [event for event in events if event.get("name") == scope]
    if not scopes:
        raise RuntimeError(f"trace has no {scope!r} collection scopes")
    def inside(event):
        start, end = event.get("ts", 0), event.get("ts", 0) + event.get("dur", 0)
        return any(event.get("pid") == block.get("pid") and event.get("tid") == block.get("tid")
                   and start >= block["ts"] and end <= block["ts"] + block.get("dur", 0) + 0.01
                   for block in scopes)
    host = [event for event in events if inside(event)]
    runtime = [event for event in host if any(token in event.get("cat", "").lower()
               for token in ("cuda_runtime", "cuda_driver")) or event.get("name", "").startswith(("cuda", "cuLaunch", "cuMemcpy", "cuStream"))]
    correlations = {_correlation(event) for event in runtime} - {None}
    external_ids = {_external(event) for event in host} - {None}
    gpu = [event for event in events if any(token in event.get("cat", "").lower()
           for token in ("kernel", "gpu_memcpy", "gpu_memset"))]
    linked = [event for event in gpu if _correlation(event) in correlations
              or _external(event) in external_ids]
    if not runtime or not gpu or not linked:
        raise RuntimeError("trace lacks correlated CUDA runtime/kernel activities; cannot verify zero staging")
    # CUDA memcpy activities without correlation still count if their device
    # timestamps overlap the scope. CPU memcpy calls themselves are not proof
    # of host transfer: actual activity direction identifies D2D vs H2D/D2H.
    temporal_transfers = [event for event in gpu if _host_transfer(event)
                          and any(block["ts"] <= event["ts"] <= block["ts"] + block.get("dur", 0)
                                  for block in scopes)]
    violations = [event for event in host if _host_wait(event) or _host_transfer(event)]
    violations += [event for event in linked if _host_transfer(event)] + temporal_transfers
    unique = {(event.get("name"), event.get("ts"), event.get("tid")): event for event in violations}
    categories = Counter(event.get("name", "") for event in linked)
    stage_us = {"physics": 0.0, "preprocess": 0.0, "task": 0.0, "other_cuda": 0.0}
    native_stages = {"rgba8_to_nchw": "preprocess", "integrate": "physics",
                     "control": "task", "finish": "task", "reset": "task"}
    for event in linked:
        stage = native_stages.get(event.get("name", ""), "other_cuda")
        stage_us[stage] += event.get("dur", 0)
    result = {"scope": scope, "scope_count": len(scopes), "cuda_runtime_calls": len(runtime),
              "linked_gpu_activities": len(linked), "host_transfers_or_waits": len(unique),
              "violations": [{"name": event.get("name"), "category": event.get("cat"),
                              "args": event.get("args", {})} for event in unique.values()],
              "cuda_stage_microseconds": stage_us, "gpu_activity_names": dict(categories),
              "render_timing": "Vulkan work is included in stream elapsed time; CUDA Kineto does not time Vulkan draws separately"}
    if unique:
        raise RuntimeError("zero-staging verification failed: " + json.dumps(result))
    return result


def capture(trainer, out, iterations=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.env.stream.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False,
                                with_stack=False) as profiler:
        for _ in range(iterations):
            length = trainer.collect()
            trainer.update(length)
        # Synchronization and profiler finalization are deliberately outside
        # sim.collect/sim.ppo_update, with trace scopes preserving that boundary.
        trainer.env.stream.synchronize()
    path = out / "trace.json"
    profiler.export_chrome_trace(str(path))
    result = {scope: verify_trace(path, scope) for scope in ("sim.collect", "sim.ppo_update")}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
