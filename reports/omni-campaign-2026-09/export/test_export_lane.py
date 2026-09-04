#!/usr/bin/env python3
"""Tests for the export + envelope lane. Stdlib only; no server, no GPU, no torch.

    python3 -m unittest discover -s <this dir> -p 'test_*.py' -v
    python3 test_export_lane.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_hf_to_gguf as export_mod  # noqa: E402
import llama_probe as probe  # noqa: E402
import parity_check as parity_mod  # noqa: E402

# --------------------------------------------------------------------------------------
# synthetic llama-server logs
# --------------------------------------------------------------------------------------

LOG_FA_Q4 = """
build: 7284 (deadbeef) with cc (Ubuntu 11.4.0) for x86_64-linux-gnu
llama_model_loader: loaded meta data with 33 key-value pairs
load_tensors: offloading 64 repeating layers to GPU
load_tensors: offloaded 65/65 layers to GPU
load_tensors:        CUDA0 model buffer size = 15079.34 MiB
llama_context: n_ctx         = 262144
llama_context: flash_attn    = enabled
llama_context: freq_base     = 5000000.0
llama_kv_cache_unified: kv_size = 262144, type_k = 'q4_0', type_v = 'q4_0', n_layer = 64
llama_kv_cache_unified:      CUDA0 KV buffer size =  4608.00 MiB
llama_context:      CUDA0 compute buffer size =   564.00 MiB
main: model loaded
main: server is listening on http://0.0.0.0:18080 - starting the main loop
srv  update_slots: all slots are idle
"""

LOG_FA_DISABLED = """
llama_context: n_ctx         = 32768
llama_context: flash_attn    = disabled
llama_kv_cache_unified: kv_size = 32768, type_k = 'f16', type_v = 'f16', n_layer = 64
llama_kv_cache_unified:      CUDA0 KV buffer size =  8192.00 MiB
main: model loaded
"""

LOG_FA_REFUSED = """
llama_context: flash_attn    = disabled
llama_context: V cache quantization requires flash_attn
llama_init_from_model: failed to initialize context
"""

LOG_FALLBACK = """
llama_context: flash_attn = auto
ggml_cuda_init: flash attention is not supported on this device, falling back to CPU
llama_kv_cache_unified: kv_size = 4096, type_k = 'f16', type_v = 'f16'
main: model loaded
"""

LOG_MULTI_GPU = """
llama_context: flash_attn = 1
llama_kv_cache_unified:      CUDA0 KV buffer size =  2304.00 MiB
llama_kv_cache_unified:      CUDA1 KV buffer size =  2304.00 MiB
load_tensors: offloaded 40/65 layers to GPU
main: server is listening
"""


class TestLogAnalysis(unittest.TestCase):
    def test_flash_attn_enabled_and_kv_types(self):
        result = probe.analyse_server_log(LOG_FA_Q4)
        self.assertEqual(result["flash_attn_state"], "enabled")
        self.assertEqual(result["kv_type_k"], "q4_0")
        self.assertEqual(result["kv_type_v"], "q4_0")
        self.assertAlmostEqual(result["kv_buffer_mib_total"], 4608.0)
        self.assertEqual(result["kv_buffer_mib_by_device"], {"CUDA0": 4608.0})
        self.assertEqual(result["layers_offloaded"], 65)
        self.assertEqual(result["layers_total"], 65)
        self.assertEqual(result["n_ctx"], 262144)
        self.assertTrue(result["server_listening"])
        self.assertTrue(result["slots_idle_seen"])
        self.assertTrue(result["model_loaded_seen"])
        self.assertEqual(result["fallback_evidence"], [])
        self.assertTrue(result["kv_quantized"])
        self.assertTrue(result["kv_quant_implies_flash_attn"])

    def test_flash_attn_disabled_f16_kv(self):
        result = probe.analyse_server_log(LOG_FA_DISABLED)
        self.assertEqual(result["flash_attn_state"], "disabled")
        self.assertFalse(result["kv_quantized"])
        self.assertFalse(result["kv_quant_implies_flash_attn"])

    def test_kv_quant_refusal_is_evidence(self):
        result = probe.analyse_server_log(LOG_FA_REFUSED)
        signals = {item["signal"] for item in result["fallback_evidence"]}
        self.assertIn("kv_quant_refused", signals)
        self.assertEqual(result["flash_attn_state"], "disabled")

    def test_cpu_fallback_line_is_captured_verbatim(self):
        result = probe.analyse_server_log(LOG_FALLBACK)
        self.assertTrue(result["fallback_evidence"])
        lines = [item["line"] for item in result["fallback_evidence"]]
        self.assertTrue(any("falling back to CPU" in line for line in lines))
        # 'auto' is recorded as auto, never silently promoted to enabled
        self.assertEqual(result["flash_attn_state"], "auto")

    def test_multi_device_kv_buffers_sum(self):
        result = probe.analyse_server_log(LOG_MULTI_GPU)
        self.assertEqual(result["flash_attn_state"], "enabled")
        self.assertAlmostEqual(result["kv_buffer_mib_total"], 4608.0)
        self.assertEqual(sorted(result["kv_buffer_mib_by_device"]), ["CUDA0", "CUDA1"])
        self.assertEqual(result["layers_offloaded"], 40)
        self.assertEqual(result["layers_total"], 65)

    def test_empty_log_is_unknown_not_a_verdict(self):
        result = probe.analyse_server_log("")
        self.assertEqual(result["flash_attn_state"], "unknown")
        self.assertIsNone(result["kv_buffer_mib_total"])
        self.assertFalse(result["kv_quant_implies_flash_attn"])


# The log the p40_256k container actually produces (llama.cpp master, 2026-09): the
# classic `llama_context: flash_attn = ...` / `KV buffer size` banner is GONE.
LOG_2026_COMPRESSED = """
warn: LLAMA_ARG_HOST environment variable is set, but will be overwritten by command line argument --host
0.00.163.912 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.402.341 I srv    load_model: loading model '/models/Qwen3.6-27B-Q4_K_M.gguf'
0.55.659.681 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 262144, kv_unified = 'false'
0.55.687.709 I srv  llama_server: model loaded
0.55.687.716 I srv  llama_server: listening on http://0.0.0.0:18080
"""


# A server log with -np 4: the 2026 build prints ONLY the per-slot context.
LOG_2026_PARALLEL = """
0.00.402.341 I srv    load_model: loading model '/models/child.gguf'
0.12.100.000 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 65536, kv_unified = 'false'
0.12.110.000 I srv  llama_server: model loaded
0.12.111.000 I srv  llama_server: listening on http://127.0.0.1:18095
"""

# The control receipt shapes `load_r3_control` returns.
# A control only speaks for the binary it was measured on, so every fixture carries the
# build join `load_r3_control()` computes. MATCHED is the ordinary case; the other two are
# the ways the join fails, and both must stop the licence.
BUILD_MATCHED = {"checked": True, "matches": True, "reasons": ["2 field(s) matched"]}
BUILD_MISMATCHED = {"checked": True, "matches": False,
                    "reasons": ["llama_cpp_commit differs: control b96806d9 != build 1111aaaa"]}
BUILD_UNJOINED = {"checked": False, "matches": None,
                  "reasons": ["no build identity was supplied"]}

R3_VERIFIED = {"path": "/x/fa-negative-control.json", "receipt_sha256": "a" * 64,
               "premise_holds": True, "observed": "refuses with -fa off",
               "refusal_line": "V cache quantization requires flash_attn",
               "build_match": BUILD_MATCHED}
R3_VOID = {"path": "/x/fa-negative-control.json", "receipt_sha256": "b" * 64,
           "premise_holds": False, "observed": "loads anyway with -fa off",
           "refusal_line": None, "build_match": BUILD_MATCHED}
R3_STALE_OTHER_BUILD = dict(R3_VERIFIED, build_match=BUILD_MISMATCHED)
R3_UNJOINED = dict(R3_VERIFIED, build_match=BUILD_UNJOINED)


# VERBATIM from the negative control run on this build (commit b96806d9), GPU2 P40,
# `-fa off -ctk q4_0 -ctv q4_0 -c 4096`. The process exited rc=1 in 3.04 s.
# Receipt: fa-control/fa-negative-control.json.
LOG_REAL_REFUSAL_2026 = """
0.00.163.912 I cmn  common_param: common_params_print_info: verbosity = 3
0.00.402.341 I srv    load_model: loading model '/home/cereal/omni-campaign/export/gguf/Q35-08B-stock.Q4_K_M.gguf'
0.00.991.944 E llama_init_from_model: quantized V cache requires flash_attn to be enabled
0.00.992.100 E srv    load_model: failed to load model
"""


class TestRealRefusalText(unittest.TestCase):
    """The R3 premise is only usable because a negative control was actually run. The
    refusal string it printed is NOT the one the parser was originally written against
    ("K/V cache quantization requires flash_attn"), so the real text is pinned here."""

    def test_parser_matches_the_refusal_this_build_actually_prints(self):
        result = probe.analyse_server_log(LOG_REAL_REFUSAL_2026)
        signals = {item["signal"] for item in result["fallback_evidence"]}
        self.assertIn("kv_quant_refused", signals)
        line = next(item["line"] for item in result["fallback_evidence"]
                    if item["signal"] == "kv_quant_refused")
        self.assertIn("quantized V cache requires flash_attn to be enabled", line)

    def test_a_refusal_voids_the_fa_reading_even_with_a_licensed_premise(self):
        flags = probe.parse_server_flags(["-fa", "off", "-ctk", "q4_0", "-ctv", "q4_0"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_REAL_REFUSAL_2026), flags,
            flags_source="own_launch_command", r3_control=R3_VERIFIED,
        )
        self.assertTrue(evidence["refusal_or_fallback_seen"])
        self.assertFalse(evidence["quantized_kv_accepted"])
        self.assertIn("did NOT run", evidence["reading"])

    def test_the_old_guessed_string_still_matches(self):
        result = probe.analyse_server_log(LOG_FA_REFUSED)
        signals = {item["signal"] for item in result["fallback_evidence"]}
        self.assertIn("kv_quant_refused", signals)


class TestCompressed2026LogFormat(unittest.TestCase):
    def test_new_format_yields_ctx_and_ready_without_a_banner(self):
        result = probe.analyse_server_log(LOG_2026_COMPRESSED)
        self.assertEqual(result["n_ctx"], 262144)
        self.assertEqual(result["kv_unified"], "false")
        self.assertTrue(result["model_loaded_seen"])
        self.assertTrue(result["server_listening"])
        # no banner at all -> the log alone knows nothing about flash attention
        self.assertEqual(result["flash_attn_state"], "unknown")
        self.assertEqual(result["fallback_evidence"], [])

    def test_n_ctx_provenance_is_recorded_not_guessed(self):
        result = probe.analyse_server_log(LOG_2026_COMPRESSED)
        self.assertEqual(result["n_ctx_slot"], 262144)
        self.assertEqual(result["n_slots"], 1)
        self.assertEqual(result["n_ctx_source"], "n_ctx_slot*n_slots")
        # an old-format log that prints a real n_ctx line uses that instead
        classic = probe.analyse_server_log(LOG_FA_Q4)
        self.assertEqual(classic["n_ctx"], 262144)
        self.assertEqual(classic["n_ctx_source"], "n_ctx")

    def test_parallel_slots_do_not_understate_the_context(self):
        # -np 4 at -c 262144 prints n_ctx_slot = 65536. Reading that AS n_ctx would report
        # a quarter of the real context on any attached server with parallel slots.
        result = probe.analyse_server_log(LOG_2026_PARALLEL)
        self.assertEqual(result["n_ctx_slot"], 65536)
        self.assertEqual(result["n_slots"], 4)
        self.assertEqual(result["n_ctx_from_slots"], 262144)
        self.assertEqual(result["n_ctx"], 262144)
        self.assertEqual(result["n_ctx_source"], "n_ctx_slot*n_slots")

    def test_log_clock_gives_the_real_load_time(self):
        # 0.55.687.716 = 55.687716 s after start; a 5 s health poll would say "60 s".
        self.assertAlmostEqual(probe.parse_log_time("0.55.687.716 I srv  x"), 55.687716, places=6)
        self.assertAlmostEqual(probe.parse_log_time("0.02.842 I srv  x"), 2.842, places=6)
        self.assertIsNone(probe.parse_log_time("main: model loaded"))
        result = probe.analyse_server_log(LOG_2026_COMPRESSED)
        self.assertAlmostEqual(result["model_loaded_at_log_s"], 55.687709, places=6)
        self.assertAlmostEqual(result["listening_at_log_s"], 55.687716, places=6)

    def test_evidence_chain_uses_launch_flags_when_the_log_is_silent(self):
        cmd = ["-m", "/models/Qwen3.6-27B-Q4_K_M.gguf", "-c", "262144", "-fa", "on",
               "-np", "1", "-ngl", "999", "-ctk", "q4_0", "-ctv", "q4_0",
               "--host", "0.0.0.0", "--port", "18080"]
        flags = probe.parse_server_flags(cmd)
        self.assertEqual(flags["flash_attn"], "on")
        self.assertEqual(flags["type_k"], "q4_0")
        self.assertEqual(flags["type_v"], "q4_0")
        self.assertEqual(flags["n_ctx"], 262144)
        self.assertEqual(flags["n_parallel"], 1)
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags,
            flags_source="docker_inspect", r3_control=R3_VERIFIED,
        )
        self.assertTrue(evidence["quantized_kv_accepted"])
        self.assertTrue(evidence["kv_pair_matched"])
        self.assertFalse(evidence["refusal_or_fallback_seen"])
        self.assertEqual(evidence["r3_premise"], "verified")
        self.assertTrue(evidence["r3_usable"])
        self.assertIn("flash-attention path ran", evidence["reading"])
        self.assertTrue(any(item.startswith("R3") for item in evidence["basis"]))

    def test_r3_without_a_control_is_labelled_unverified_not_asserted(self):
        flags = probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags,
            flags_source="own_launch_command", r3_control=None,
        )
        self.assertEqual(evidence["r3_premise"], "unverified_on_this_build")
        self.assertFalse(evidence["r3_usable"])
        self.assertIn("UNVERIFIED", evidence["reading"])
        self.assertNotIn("flash-attention path ran", evidence["reading"])

    def test_r3_is_voided_when_the_control_says_the_build_loads_anyway(self):
        flags = probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags,
            flags_source="own_launch_command", r3_control=R3_VOID,
        )
        self.assertEqual(evidence["r3_premise"], "void")
        self.assertFalse(evidence["r3_usable"])
        self.assertIn("R3 IS VOID", evidence["reading"])

    def test_unobserved_cli_defaults_never_become_evidence(self):
        # --attach with no --container-name: `requested` is our own CLI defaults. A receipt
        # that consumed them would assert a q4_0 FA run against a server it never inspected.
        typed = {"flash_attn": "on", "type_k": "q4_0", "type_v": "q4_0", "n_ctx": 262144}
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), typed,
            flags_source="cli_defaults_unverified", r3_control=R3_VERIFIED,
        )
        self.assertFalse(evidence["flags_observed"])
        self.assertIsNone(evidence["requested_flash_attn"])
        self.assertIsNone(evidence["requested_type_k"])
        self.assertFalse(evidence["quantized_kv_accepted"])
        self.assertFalse(evidence["r3_usable"])
        self.assertNotIn("flash-attention path ran", evidence["reading"])
        self.assertTrue(any(item.startswith("R0") for item in evidence["basis"]))

    def test_empty_docker_inspect_leaves_no_flags_to_assert(self):
        # the concrete failure path: container renamed / docker unavailable -> inspect
        # returns [], parse_server_flags({}) is empty, and the caller falls back to defaults
        self.assertEqual(probe.parse_server_flags([]), {})
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), {},
            flags_source="cli_defaults_unverified", r3_control=R3_VERIFIED,
        )
        self.assertFalse(evidence["quantized_kv_accepted"])
        self.assertIn("no direct evidence", evidence["reading"])

    def test_evidence_chain_reports_no_evidence_for_f16_kv(self):
        flags = probe.parse_server_flags(["-c", "4096", "-ctk", "f16", "-ctv", "f16"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags,
            flags_source="docker_inspect", r3_control=R3_VERIFIED,
        )
        self.assertFalse(evidence["quantized_kv_accepted"])
        self.assertIn("no direct evidence", evidence["reading"])

    def test_refusal_overrides_everything(self):
        flags = probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_FA_REFUSED), flags,
            flags_source="own_launch_command", r3_control=R3_VERIFIED,
        )
        self.assertTrue(evidence["refusal_or_fallback_seen"])
        self.assertIn("did NOT run", evidence["reading"])

    def test_mismatched_kv_pair_is_flagged(self):
        flags = probe.parse_server_flags(["-ctk", "q8_0", "-ctv", "q4_0"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags,
            flags_source="docker_inspect", r3_control=R3_VERIFIED,
        )
        self.assertFalse(evidence["kv_pair_matched"])

    def test_bare_fa_flag_is_recorded_as_implicit(self):
        flags = probe.parse_server_flags("-m /x.gguf -fa -c 8192")
        self.assertEqual(flags["flash_attn"], "on_implicit")
        self.assertEqual(flags["n_ctx"], 8192)

    def test_long_flag_aliases(self):
        flags = probe.parse_server_flags(
            ["--flash-attn", "auto", "--cache-type-k", "q8_0", "--ctx-size", "65536",
             "--parallel", "1", "--gpu-layers", "999"]
        )
        self.assertEqual(flags["flash_attn"], "auto")
        self.assertEqual(flags["type_k"], "q8_0")
        self.assertEqual(flags["n_ctx"], 65536)
        self.assertEqual(flags["n_gpu_layers"], 999)

    def test_empty_command_is_empty_flags(self):
        self.assertEqual(probe.parse_server_flags(None), {})
        self.assertEqual(probe.parse_server_flags([]), {})


class TestNvidiaSmiParse(unittest.TestCase):
    def test_parse_used_and_total(self):
        text = "0, 21558, 24576\n1, 5, 24576\n6, 46000, 46080\n"
        rows = probe.parse_nvidia_smi_csv(text)
        self.assertEqual(rows[0]["memory_used_mib"], 21558.0)
        self.assertEqual(rows[6]["memory_total_mib"], 46080.0)
        self.assertEqual(sorted(rows), [0, 1, 6])

    def test_garbage_lines_are_ignored(self):
        rows = probe.parse_nvidia_smi_csv("no such device\n\n2, 7, 24576\n")
        self.assertEqual(list(rows), [2])


class TestComputeAppsParse(unittest.TestCase):
    """Per-PROCESS VRAM. Whole-card peak includes every other tenant of the card, so a
    run's own share has to be measured here, not derived by subtracting a remembered
    number from the whole-card peak."""

    SAMPLE = (
        "GPU-2ce6a95c-7b1f-644a-3808-d781732a0582, 999888777, "
        "/home/cereal/omni-campaign/export/llamacpp/bin/llama-server, 1871\n"
        "GPU-2ce6a95c-7b1f-644a-3808-d781732a0582, 999001, python, 630\n"
    )

    def test_rows_carry_pid_uuid_and_mib(self):
        rows = probe.parse_compute_apps_csv(self.SAMPLE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["pid"], 999888777)
        self.assertEqual(rows[0]["memory_used_mib"], 1871.0)
        self.assertTrue(rows[0]["process_name"].endswith("llama-server"))
        self.assertEqual(rows[1]["memory_used_mib"], 630.0)

    def test_empty_and_na_are_not_rows(self):
        self.assertEqual(probe.parse_compute_apps_csv(""), [])
        self.assertEqual(probe.parse_compute_apps_csv("[N/A]\n"), [])
        self.assertEqual(probe.parse_compute_apps_csv("garbage,line\n"), [])

    def test_sampler_reports_own_share_separately_from_whole_card(self):
        # indices=[-1] matches no real GPU, so stop()'s final poll (which really does run
        # nvidia-smi when one exists) cannot contaminate the synthetic rows below.
        sampler = probe.GpuSampler(interval=99.0, indices=[-1], track_pids=[999888777])
        sampler.uuid_to_index["GPU-2ce6a95c-7b1f-644a-3808-d781732a0582"] = 4
        for row in probe.parse_compute_apps_csv(self.SAMPLE):
            index = sampler.uuid_to_index.get(row["gpu_uuid"])
            slot = sampler.peak_by_pid.setdefault(str(row["pid"]), {
                "peak_memory_used_mib": 0.0,
                "process_name": row["process_name"],
                "gpu_uuid": row["gpu_uuid"],
                "gpu_index": index,
                "tracked_as_ours": row["pid"] in sampler.track_pids,
            })
            slot["peak_memory_used_mib"] = max(slot["peak_memory_used_mib"], row["memory_used_mib"])
        sampler.peak[4] = {"peak_memory_used_mib": 2501.0, "memory_total_mib": 24576.0}
        report = sampler.stop()
        self.assertEqual(report["peak_by_index"]["4"]["peak_memory_used_mib"], 2501.0)
        self.assertEqual(report["own_peak_memory_used_mib"], 1871.0)
        self.assertEqual(report["tracked_pids"], ["999888777"])
        self.assertTrue(report["peak_by_pid"]["999888777"]["tracked_as_ours"])
        self.assertFalse(report["peak_by_pid"]["999001"]["tracked_as_ours"])


class TestNeedle(unittest.TestCase):
    def test_needle_appears_once_at_requested_depth(self):
        prompt, insert_at = probe.build_needle_prompt(100, 0.5, "ZQ7-4418")
        self.assertEqual(prompt.count("ZQ7-4418"), 1)
        self.assertEqual(insert_at, 50)  # round(0.5 * 99) == 50
        self.assertTrue(prompt.rstrip().endswith("Answer:"))
        self.assertIn("Line 000100", prompt)

    def test_depth_extremes_are_exact(self):
        head, at_head = probe.build_needle_prompt(50, 0.0, "AAA-111")
        tail, at_tail = probe.build_needle_prompt(50, 1.0, "AAA-111")
        self.assertEqual(at_head, 0)
        self.assertEqual(at_tail, 50)  # after ALL 50 filler lines, i.e. genuinely last
        self.assertLess(head.index("AAA-111"), head.index("Line 000001"))
        self.assertGreater(tail.index("AAA-111"), tail.index("Line 000050"))
        # depth 1.0 used to land second-to-last; assert NO haystack line follows the needle
        self.assertNotIn("Line 0", tail.split("AAA-111")[1])
        # and at depth 0.0 no haystack line precedes it
        self.assertNotIn("Line 0", head.split("AAA-111")[0])

    def test_needle_is_an_extra_line_so_filler_count_is_preserved(self):
        for depth in (0.0, 0.25, 0.5, 0.75, 1.0):
            prompt, _ = probe.build_needle_prompt(20, depth, "BBB-222")
            self.assertEqual(prompt.count("BBB-222"), 1)
            for index in (1, 20):
                self.assertIn("Line %06d" % index, prompt)

    def test_bad_arguments_rejected(self):
        with self.assertRaises(ValueError):
            probe.build_needle_prompt(0, 0.5, "X")
        with self.assertRaises(ValueError):
            probe.build_needle_prompt(10, 1.5, "X")

    def test_scoring_is_case_and_whitespace_tolerant(self):
        self.assertTrue(probe.score_needle("  the code is zq7-4418.\n", "ZQ7-4418")["hit"])
        self.assertTrue(probe.score_needle("ZQ7-4418", "ZQ7-4418")["hit"])
        self.assertFalse(probe.score_needle("I do not know.", "ZQ7-4418")["hit"])
        self.assertFalse(probe.score_needle("", "ZQ7-4418")["hit"])
        self.assertFalse(probe.score_needle(None, "ZQ7-4418")["hit"])

    def test_haystack_line_numbering(self):
        text = probe.build_haystack(3)
        self.assertEqual(text.count("\n"), 3)
        self.assertIn("Line 000001", text)
        self.assertIn("Line 000003", text)


class TestCalibration(unittest.TestCase):
    class FakeServer(object):
        """Tokenizes at a fixed tokens-per-character rate."""

        def __init__(self, chars_per_token=4.0):
            self.chars_per_token = chars_per_token
            self.calls = 0

        def token_count(self, text, timeout=None):
            self.calls += 1
            return int(len(text) / self.chars_per_token)

    def test_converges_within_tolerance(self):
        server = self.FakeServer()
        result = probe.calibrate_line_count(server, 4096, "ZQ7-4418", tolerance=0.02)
        self.assertLessEqual(abs(result["tokens"] - 4096), 0.02 * 4096)
        self.assertGreaterEqual(result["lines"], 4)
        self.assertEqual(result["prompt"].count("ZQ7-4418"), 1)
        self.assertLessEqual(server.calls, 6)

    def test_records_every_round(self):
        server = self.FakeServer(chars_per_token=9.0)
        result = probe.calibrate_line_count(server, 32768, "MB3-9052")
        self.assertTrue(result["rounds"])
        self.assertEqual(result["rounds"][-1]["tokens"], result["tokens"])


class TestContextHeadroom(unittest.TestCase):
    """The 256k point must not ask for a prompt the server will refuse.

    The sibling lane's real needle log has two rows that read
    `{"depth_target": 250000, "error": "<HTTPError 400: 'Bad Request'>"}` against a
    `-c 262144` server: the prompt plus the requested generation did not fit the slot.
    These tests pin the fix, so the marquee 262,144-token point cannot come back as a
    400 again.
    """

    class CeilingServer(object):
        """Tokenizer at a fixed rate that also REFUSES like llama-server does.

        `completion` raises when prompt_tokens + n_predict > n_ctx, which is the real
        behaviour the calibration has to stay under.
        """

        def __init__(self, chars_per_token=4.0, n_ctx=262144):
            self.chars_per_token = chars_per_token
            self.n_ctx = n_ctx
            self.refusals = 0
            self.accepted = []

        def token_count(self, text, timeout=None):
            return int(len(text) / self.chars_per_token)

        def completion(self, prompt, n_predict=64, cache_prompt=False, seed=1234,
                       timeout=None, extra=None):
            tokens = self.token_count(prompt)
            if tokens + int(n_predict) > self.n_ctx:
                self.refusals += 1
                raise urllib.error.HTTPError(
                    "http://127.0.0.1:18090/completion", 400, "Bad Request", {}, None
                )
            self.accepted.append((tokens, int(n_predict)))
            return {"content": " ZQ7-4418", "prompt_n": tokens, "wall_s": 0.1,
                    "prompt_per_second": 500.0, "predicted_per_second": 20.0}

    def test_headroom_reserves_the_largest_generation_plus_safety(self):
        room = probe.context_point_headroom(262144, 64)
        self.assertEqual(room["n_ctx"], 262144)
        # 64 decode tokens is bigger than the 32 needle tokens, plus the 64-token margin
        self.assertEqual(room["reserve_tokens"], 64 + probe.CONTEXT_SAFETY_TOKENS)
        self.assertEqual(room["prompt_ceiling"], 262144 - room["reserve_tokens"])
        # a needle-only budget still reserves the needle's own n_predict
        self.assertEqual(probe.context_point_headroom(4096, 1)["reserve_tokens"],
                         probe.NEEDLE_PREDICT_TOKENS + probe.CONTEXT_SAFETY_TOKENS)
        self.assertIsNone(probe.context_point_headroom(None, 64)["prompt_ceiling"])
        with self.assertRaises(ValueError):
            probe.context_point_headroom(64, 64)

    def test_calibration_never_returns_a_prompt_over_the_ceiling(self):
        server = self.CeilingServer()
        ceiling = probe.context_point_headroom(262144, 64)["prompt_ceiling"]
        result = probe.calibrate_line_count(server, 262144, "ZQ7-4418", max_tokens=ceiling)
        self.assertLessEqual(result["tokens"], ceiling)
        self.assertFalse(result["over_cap"])
        self.assertTrue(result["capped"])
        self.assertEqual(result["requested_target_tokens"], 262144)
        self.assertEqual(result["target_tokens"], ceiling)
        # and it is still close to the ceiling: a cap must not silently shrink the point
        self.assertGreater(result["tokens"], ceiling * 0.95)

    def test_uncapped_calibration_is_unchanged(self):
        server = self.CeilingServer()
        result = probe.calibrate_line_count(server, 32768, "MB3-9052")
        self.assertLessEqual(abs(result["tokens"] - 32768), 0.02 * 32768)
        self.assertFalse(result["capped"])
        self.assertIsNone(result["max_tokens"])

    def test_the_full_context_point_is_accepted_by_a_refusing_server(self):
        """End to end: the 262,144 point against a server that 400s on overflow."""
        import envelope_bench  # noqa: PLC0415

        server = self.CeilingServer()
        point = envelope_bench.measure_context_point(
            server, 262144, "ZQ7-4418", 64, [0.1, 0.9], do_needle=True,
            request_timeout=30.0, n_ctx=262144,
        )
        self.assertEqual(server.refusals, 0, "the calibrated prompt was refused")
        self.assertTrue(point["capped_by_ctx"])
        self.assertIsNone(point.get("error"))
        self.assertEqual(len(point["needle"]), 2)
        self.assertTrue(all(item["hit"] for item in point["needle"]))
        for tokens, n_predict in server.accepted:
            self.assertLessEqual(tokens + n_predict, 262144)

    def test_without_n_ctx_the_old_overflow_still_happens_but_is_recorded_not_raised(self):
        """No n_ctx given -> no ceiling -> the server refuses; the sweep must survive."""
        import envelope_bench  # noqa: PLC0415

        server = self.CeilingServer()
        point = envelope_bench.measure_context_point(
            server, 262144, "ZQ7-4418", 64, [0.1], do_needle=True,
            request_timeout=30.0, n_ctx=None,
        )
        self.assertGreaterEqual(server.refusals, 1)
        self.assertIn("HTTPError", point["error"])
        self.assertEqual(point["needle"], [])

    def test_needle_refusal_is_recorded_per_depth(self):
        import envelope_bench  # noqa: PLC0415

        class NeedleRefuser(TestContextHeadroom.CeilingServer):
            def completion(self, prompt, n_predict=64, **kwargs):
                if int(n_predict) == probe.NEEDLE_PREDICT_TOKENS:
                    self.refusals += 1
                    raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
                return super().completion(prompt, n_predict=n_predict, **kwargs)

        server = NeedleRefuser()
        point = envelope_bench.measure_context_point(
            server, 4096, "ZQ7-4418", 64, [0.1, 0.9], do_needle=True,
            request_timeout=30.0, n_ctx=262144,
        )
        self.assertIsNone(point.get("error"))
        self.assertEqual(len(point["needle"]), 2)
        for item in point["needle"]:
            self.assertIn("HTTPError", item["error"])
            self.assertFalse(item["hit"])


class TestReceipts(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self):
        left = probe.canonical_json_bytes({"b": 1, "a": [3, 2]})
        right = probe.canonical_json_bytes({"a": [3, 2], "b": 1})
        self.assertEqual(left, right)
        self.assertTrue(left.endswith(b"\n"))

    def test_write_receipt_is_atomic_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "receipt.json")
            digest = probe.write_receipt(path, {"kind": "test", "value": 2})
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))
            with open(path, "rb") as handle:
                blob = handle.read()
            self.assertEqual(digest, probe.sha256_bytes(blob))
            payload = json.loads(blob)
            self.assertEqual(payload["schema"], probe.SCHEMA)
            self.assertEqual(payload["value"], 2)

    def test_sha256_file_matches_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blob.bin")
            data = os.urandom(1024 * 33)
            with open(path, "wb") as handle:
                handle.write(data)
            self.assertEqual(probe.sha256_file(path), probe.sha256_bytes(data))

    def test_directory_digest_is_order_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "sub"))
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                handle.write('{"a": 1}')
            with open(os.path.join(tmp, "sub", "b.json"), "w", encoding="utf-8") as handle:
                handle.write('{"b": 2}')
            with open(os.path.join(tmp, "weights.bin"), "wb") as handle:
                handle.write(b"\x00" * 8)
            with open(os.path.join(tmp, "notes.md"), "w", encoding="utf-8") as handle:
                handle.write("not part of the checkpoint")
            first = probe.directory_digest(tmp)
            # .bin IS covered now: the first version of this allowlist skipped every weight
            # container, so a checkpoint sharded as pytorch_model-*.bin digested to a hash
            # over its config and tokenizer alone and two different sets of weights under
            # one config collided on the lineage key.
            self.assertEqual(first["file_count"], 3)
            self.assertIn("weights.bin", [item["path"] for item in first["files"]])
            self.assertEqual(first["skipped_files"], ["notes.md"])
            self.assertIn(".bin", first["suffix_filter"])
            second = probe.directory_digest(tmp)
            self.assertEqual(first["directory_sha256"], second["directory_sha256"])
            with open(os.path.join(tmp, "sub", "b.json"), "w", encoding="utf-8") as handle:
                handle.write('{"b": 3}')
            third = probe.directory_digest(tmp)
            self.assertNotEqual(first["directory_sha256"], third["directory_sha256"])


class TestServerPayload(unittest.TestCase):
    def test_greedy_payload_is_deterministic(self):
        payload = probe.LlamaServer.greedy_payload("hi", 16)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["repeat_penalty"], 1.0)
        self.assertFalse(payload["cache_prompt"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["n_predict"], 16)

    def test_summarize_completion_extracts_timings(self):
        body = {
            "content": " 1081",
            "stop_type": "limit",
            "tokens_predicted": 4,
            "timings": {"prompt_n": 4096, "prompt_ms": 8000.0, "prompt_per_second": 512.0,
                        "predicted_n": 4, "predicted_ms": 260.0, "predicted_per_second": 15.4},
        }
        flat = probe.summarize_completion(body, 8.5)
        self.assertEqual(flat["prompt_n"], 4096)
        self.assertEqual(flat["prompt_per_second"], 512.0)
        self.assertEqual(flat["prompt_ms"], 8000.0)
        self.assertEqual(flat["predicted_per_second"], 15.4)
        self.assertEqual(flat["wall_s"], 8.5)
        self.assertEqual(flat["content"], " 1081")

    def test_summarize_completion_tolerates_missing_timings(self):
        flat = probe.summarize_completion({"content": "x"}, 1.0)
        self.assertIsNone(flat["prompt_per_second"])
        self.assertEqual(flat["content"], "x")


class TestPropsAndLogHelpers(unittest.TestCase):
    def test_props_n_ctx_top_level(self):
        self.assertEqual(probe.props_n_ctx({"n_ctx": 262144}), 262144)

    def test_props_n_ctx_nested(self):
        self.assertEqual(
            probe.props_n_ctx({"default_generation_settings": {"n_ctx": 32768}}), 32768
        )
        self.assertEqual(
            probe.props_n_ctx({"default_generation_settings": {"params": {"n_ctx": 4096}}}), 4096
        )

    def test_props_n_ctx_absent(self):
        self.assertIsNone(probe.props_n_ctx({}))
        self.assertIsNone(probe.props_n_ctx({"default_generation_settings": {"params": {}}}))
        self.assertIsNone(probe.props_n_ctx("not a dict"))

    def test_docker_logs_reports_failure_instead_of_raising(self):
        text = probe.docker_logs("oc-no-such-container-xyz", tail=5, timeout=60)
        self.assertIsInstance(text, str)
        # either the rc marker (docker present) or empty-ish output (docker absent)
        self.assertTrue(text == "" or "rc=" in text or "No such container" in text)


class TestListParsing(unittest.TestCase):
    def test_int_list_sorted_deduped(self):
        self.assertEqual(probe.parse_int_list("262144, 4096,32768,4096"), [4096, 32768, 262144])

    def test_int_list_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            probe.parse_int_list("4096,0")
        with self.assertRaises(ValueError):
            probe.parse_int_list("-1")

    def test_float_list_range_checked(self):
        self.assertEqual(probe.parse_float_list("0.9,0.1,0.5"), [0.1, 0.5, 0.9])
        with self.assertRaises(ValueError):
            probe.parse_float_list("1.5")


class TestExportPlan(unittest.TestCase):
    def test_quant_parsing_and_order(self):
        self.assertEqual(export_mod.parse_quants("Q4_K_M,IQ4_XS,q4_k_m"), ["Q4_K_M", "IQ4_XS"])

    def test_unknown_quant_rejected(self):
        with self.assertRaises(ValueError):
            export_mod.parse_quants("Q4_K_M,Q2_TOTALLY_MADE_UP")

    def test_tag_is_filename_safe(self):
        self.assertEqual(export_mod.safe_tag("OC01-F-Q38U-A1"), "OC01-F-Q38U-A1")
        for bad in ("../escape", "a/b", "", "..", "-rf"):
            with self.assertRaises(ValueError):
                export_mod.safe_tag(bad)

    def test_gguf_naming(self):
        self.assertEqual(export_mod.gguf_name("OC01", "bf16"), "OC01.bf16.gguf")
        self.assertEqual(export_mod.gguf_name("OC01", "Q4_K_M"), "OC01.Q4_K_M.gguf")

    def test_read_hf_arch_flat_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "tie_word_embeddings": False,
                "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
            }
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            read = export_mod.read_hf_arch(tmp)
            self.assertEqual(read["model_type"], "qwen3_5")
            self.assertEqual(read["layer_types_len"], 64)
            self.assertEqual(read["vocab_size"], 248320)
            self.assertEqual(read["resolved_from"]["vocab_size"], "config.json")

    def test_read_hf_arch_resolves_nested_sub_config(self):
        # the shape that shipped a receipt full of nulls: everything lives in text_config
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {
                    "num_hidden_layers": 24,
                    "hidden_size": 1024,
                    "vocab_size": 248320,
                    "max_position_embeddings": 262144,
                    "tie_word_embeddings": True,
                    "head_dim": 256,
                    "attn_output_gate": True,
                    "full_attention_interval": 4,
                    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
                },
                "vision_config": {"hidden_size": 768},
            }
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            read = export_mod.read_hf_arch(tmp)
            self.assertEqual(read["num_hidden_layers"], 24)
            self.assertEqual(read["hidden_size"], 1024)      # text tower, not vision
            self.assertEqual(read["vocab_size"], 248320)
            self.assertEqual(read["max_position_embeddings"], 262144)
            self.assertEqual(read["layer_types_len"], 24)
            self.assertEqual(read["resolved_from"]["hidden_size"], "text_config")
            self.assertEqual(read["resolved"]["head_dim"], 256)
            self.assertEqual(read["resolved"]["attn_output_gate"], True)
            self.assertEqual(read["resolved"]["full_attention_interval"], 4)
            self.assertEqual(read["sub_configs_present"], ["text_config"])
            self.assertEqual(read["warnings"], [])
            # the top-level values are kept so a disagreement stays visible
            self.assertIsNone(read["top_level"]["hidden_size"])

    def test_read_hf_arch_pinned_to_the_real_checkpoint_config(self):
        # byte-identical copy of /home/cereal/models/Qwen3.5-0.8B/config.json, the
        # checkpoint this lane actually converted (sha256 verified against the server).
        fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        fixture = os.path.join(fixture_dir, "Qwen3.5-0.8B.config.json")
        self.assertTrue(os.path.exists(fixture), "missing pinned fixture %s" % fixture)
        self.assertEqual(
            probe.sha256_file(fixture),
            "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with open(fixture, "rb") as src, open(os.path.join(tmp, "config.json"), "wb") as dst:
                dst.write(src.read())
            read = export_mod.read_hf_arch(tmp)
            self.assertEqual(read["architectures"], ["Qwen3_5ForConditionalGeneration"])
            self.assertEqual(read["num_hidden_layers"], 24)
            self.assertEqual(read["hidden_size"], 1024)
            self.assertEqual(read["vocab_size"], 248320)
            self.assertEqual(read["max_position_embeddings"], 262144)
            self.assertEqual(read["layer_types_len"], 24)
            self.assertEqual(
                read["layer_types_counts"], {"full_attention": 6, "linear_attention": 18}
            )
            self.assertEqual(read["resolved"]["head_dim"], 256)
            self.assertEqual(read["resolved"]["attn_output_gate"], True)
            self.assertEqual(read["resolved"]["full_attention_interval"], 4)
            self.assertEqual(read["resolved"]["num_key_value_heads"], 2)
            self.assertEqual(read["config_sha256"], probe.sha256_file(fixture))
            self.assertEqual(read["warnings"], [])

    def test_read_hf_arch_warns_loudly_instead_of_writing_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"architectures": ["MysteryModel"], "model_type": "mystery"}, handle)
            read = export_mod.read_hf_arch(tmp)
            self.assertIsNone(read["hidden_size"])
            self.assertEqual(len(read["warnings"]), len(export_mod.REQUIRED_ARCH_FIELDS))
            self.assertTrue(all("REQUIRED" in item for item in read["warnings"]))

    def test_dry_run_plan_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            rc = export_mod.main([
                "--hf-dir", tmp, "--out-dir", out, "--tag", "T1", "--dry-run",
                "--quants", "Q4_K_M",
            ])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(out))

    def test_missing_tools_exit_3_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            rc = export_mod.main([
                "--hf-dir", tmp, "--out-dir", out, "--tag", "T1",
                "--llamacpp-root", os.path.join(tmp, "nope"), "--skip-digest",
            ])
            self.assertEqual(rc, 3)


class TestParityCompare(unittest.TestCase):
    def test_exact_and_normalized(self):
        result = parity_mod.compare_pair("42", "42")
        self.assertTrue(result["exact_match"])
        self.assertTrue(result["normalized_match"])
        result = parity_mod.compare_pair(" 42 \n", "42")
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["normalized_match"])

    def test_char_prefix_measures_divergence_point(self):
        result = parity_mod.compare_pair("The answer is 1081.", "The answer is 1080.")
        self.assertFalse(result["normalized_match"])
        self.assertEqual(result["char_prefix"], len("The answer is 108"))

    def test_token_prefix_uses_supplied_tokenizer(self):
        class FakeTokenizer(object):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(char) for char in text]}

        result = parity_mod.compare_pair("abcd", "abzz", FakeTokenizer())
        self.assertEqual(result["token_prefix"], 2)
        self.assertEqual(result["token_len_gguf"], 4)

    def test_bank_loads_and_has_fifteen_prompts(self):
        bank = parity_mod.load_bank(parity_mod.DEFAULT_BANK)
        self.assertEqual(bank["bank_id"], "EXPORT-PB1")
        self.assertEqual(len(bank["prompts"]), 15)
        ids = [item["id"] for item in bank["prompts"]]
        self.assertEqual(len(set(ids)), 15)
        for item in bank["prompts"]:
            self.assertTrue(item["prompt"].strip())
            self.assertGreater(item["n_predict"], 0)


class TestEnvelopeSafety(unittest.TestCase):
    def test_handle_never_stops_what_it_did_not_start(self):
        import envelope_bench  # noqa: PLC0415

        handle = envelope_bench.ServerHandle()
        handle.mode = "attach"
        handle.container = "p40_256k"
        result = handle.stop()
        self.assertFalse(result["stopped"])
        self.assertIn("not started", result["reason"])

    def test_server_args_matched_kv_pair_and_np1(self):
        import envelope_bench  # noqa: PLC0415

        args = envelope_bench.server_args("/m.gguf", 262144, "q4_0", 18090, "on", None)
        self.assertIn("-ctk", args)
        self.assertEqual(args[args.index("-ctk") + 1], "q4_0")
        self.assertEqual(args[args.index("-ctv") + 1], "q4_0")
        self.assertEqual(args[args.index("-np") + 1], "1")
        self.assertEqual(args[args.index("-fa") + 1], "on")
        self.assertEqual(args[args.index("-c") + 1], "262144")

    def test_server_binds_loopback_by_default(self):
        # llama-server has no auth and warns about it in its own log; this box also fronts
        # Keycloak/mailcow/the SOC dashboard, so a multi-hour sweep must not listen on 0.0.0.0
        import envelope_bench  # noqa: PLC0415

        args = envelope_bench.server_args("/m.gguf", 4096, "q4_0", 18090, "on", None)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        override = envelope_bench.server_args("/m.gguf", 4096, "q4_0", 1, "on", None,
                                              bind="0.0.0.0")
        self.assertEqual(override[override.index("--host") + 1], "0.0.0.0")

    def test_fa_negative_control_binds_loopback_too(self):
        import fa_negative_control  # noqa: PLC0415

        with open(fa_negative_control.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('"--host", "127.0.0.1"', source)
        self.assertNotIn('"0.0.0.0"', source)

    def test_extra_args_are_appended(self):
        import envelope_bench  # noqa: PLC0415

        args = envelope_bench.server_args("/m.gguf", 4096, "f16", 1, "off", "--no-mmap --verbose")
        self.assertIn("--no-mmap", args)
        self.assertIn("--verbose", args)

    def test_context_point_above_ctx_is_rejected(self):
        import envelope_bench  # noqa: PLC0415

        with self.assertRaises(SystemExit):
            envelope_bench.main([
                "--tag", "T", "--attach", "http://127.0.0.1:1", "--ctx", "4096",
                "--prefill-points", "8192", "--needle-points", "8192",
            ])


LIFECYCLE_VERBS = {"stop", "kill", "rm", "restart", "pause", "unpause", "start", "update",
                   "commit", "prune"}


def docker_lifecycle_calls(path):
    """Every docker invocation in `path` whose verb changes container state.

    A literal grep for '"stop"' misses `docker restart`, a verb built from a variable, and
    a name built by format(). This walks the AST instead: any list/tuple literal or string
    that starts a docker command is checked against LIFECYCLE_VERBS, and a non-constant
    verb is reported too (it cannot be proven safe).
    """
    import ast  # noqa: PLC0415

    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    findings = []
    for node in ast.walk(tree):
        tokens = None
        if isinstance(node, (ast.List, ast.Tuple)):
            tokens = list(node.elts)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if text.startswith("docker ") or " docker " in text:
                words = text.replace("|", " ").split()
                for index, word in enumerate(words):
                    if word == "docker" and index + 1 < len(words):
                        if words[index + 1] in LIFECYCLE_VERBS:
                            findings.append("string: %s" % text[:80])
            continue
        if not tokens:
            continue
        first = tokens[0]
        if not (isinstance(first, ast.Constant) and first.value == "docker"):
            continue
        for element in tokens[1:]:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                if element.value.lstrip("-") in LIFECYCLE_VERBS:
                    findings.append("list verb: %s" % element.value)
                if not element.value.startswith("-"):
                    break
            else:
                findings.append("non-constant docker argument at line %d" % node.lineno)
                break
    return findings


class TestProbeReadOnly(unittest.TestCase):
    def test_module_contains_no_container_lifecycle_calls(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p40probe_check.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in ('"stop"', '"kill"', '"rm"', "docker stop", "docker kill", "docker rm"):
            self.assertNotIn(forbidden, source, "p40probe_check.py must stay read-only")

    def test_ast_finds_no_lifecycle_verb_including_restart(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("p40probe_check.py", "llama_probe.py", "parity_check.py"):
            self.assertEqual(
                docker_lifecycle_calls(os.path.join(here, name)), [],
                "%s must not change container state" % name,
            )

    def test_ast_check_actually_catches_a_lifecycle_call(self):
        # the detector is worthless if it cannot see a real one: envelope_bench.py DOES
        # stop containers (only ones it created), so it must be reported.
        here = os.path.dirname(os.path.abspath(__file__))
        findings = docker_lifecycle_calls(os.path.join(here, "envelope_bench.py"))
        self.assertTrue(findings, "detector failed to see envelope_bench's docker rm -f")
        self.assertTrue(any("rm" in item for item in findings))

    def test_ast_check_catches_a_synthetic_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sneaky.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('verb = "restart"\nrun(["docker", "restart", name])\n')
            self.assertTrue(any("restart" in item for item in docker_lifecycle_calls(path)))


class TestDigestCoversWeights(unittest.TestCase):
    def test_changing_a_weight_file_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                handle.write('{"a": 1}')
            weights = os.path.join(tmp, "pytorch_model-00001-of-00002.bin")
            with open(weights, "wb") as handle:
                handle.write(b"\x01" * 64)
            before = probe.directory_digest(tmp)["directory_sha256"]
            with open(weights, "wb") as handle:
                handle.write(b"\x02" * 64)
            after = probe.directory_digest(tmp)["directory_sha256"]
            self.assertNotEqual(before, after)

    def test_suffix_filter_none_digests_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "x.weird"), "wb") as handle:
                handle.write(b"z")
            result = probe.directory_digest(tmp, suffixes=None)
            self.assertEqual(result["file_count"], 1)
            self.assertIsNone(result["suffix_filter"])
            self.assertEqual(result["skipped_file_count"], 0)


class TestReceiptVersioning(unittest.TestCase):
    """A receipt hash cited by another receipt must stay resolvable to bytes.

    The lane shipped an R3 control naming build receipt c1ab3f9d while the file on disk
    hashed to 4a9554fe: write_receipt os.replace()d it in place and the cited bytes
    survived only as a hand-made .bak.
    """

    def test_sidecar_is_named_after_its_own_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            digest = probe.write_receipt(path, {"kind": "test", "n": 1})
            sidecar = os.path.join(tmp, "r.%s.json" % digest[:8])
            self.assertTrue(os.path.exists(sidecar))
            with open(sidecar, "rb") as handle:
                self.assertEqual(probe.sha256_bytes(handle.read()), digest)

    def test_a_rewrite_leaves_the_old_bytes_resolvable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            first = probe.write_receipt(path, {"kind": "test", "n": 1})
            second = probe.write_receipt(path, {"kind": "test", "n": 2})
            self.assertNotEqual(first, second)
            old_sidecar = os.path.join(tmp, "r.%s.json" % first[:8])
            self.assertTrue(os.path.exists(old_sidecar))
            with open(old_sidecar, "rb") as handle:
                self.assertEqual(json.loads(handle.read())["n"], 1)
            with open(path, "rb") as handle:
                self.assertEqual(json.loads(handle.read())["n"], 2)

    def test_versioning_can_be_turned_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            digest = probe.write_receipt(path, {"kind": "test"}, keep_versions=False)
            self.assertFalse(os.path.exists(os.path.join(tmp, "r.%s.json" % digest[:8])))


class TestR3BuildJoin(unittest.TestCase):
    """A control measured on one binary must not license another one."""

    def _fake_build(self, tmp, commit, server_bytes):
        root = os.path.join(tmp, "llamacpp")
        os.makedirs(os.path.join(root, "bin"))
        binary = os.path.join(root, "bin", "llama-server")
        with open(binary, "wb") as handle:
            handle.write(server_bytes)
        probe.write_receipt(os.path.join(root, "build-receipt.json"), {
            "kind": "llamacpp_build",
            "llama_cpp_commit": commit,
            "binaries": {"llama-server": {"sha256": probe.sha256_file(binary)}},
        })
        return root

    def test_build_identity_reads_commit_and_binary_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_build(tmp, "b96806d9", b"BINARY-A")
            identity = probe.build_identity(llamacpp_root=root)
            self.assertEqual(identity["llama_cpp_commit"], "b96806d9")
            self.assertEqual(identity["llama_server_sha256"],
                             probe.sha256_bytes(b"BINARY-A"))
            self.assertEqual(identity["llama_server_sha256_on_disk"],
                             identity["llama_server_sha256"])

    def test_build_identity_is_none_when_there_is_no_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(probe.build_identity(llamacpp_root=os.path.join(tmp, "nope")))

    def test_control_from_the_same_build_joins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_build(tmp, "b96806d9", b"BINARY-A")
            control = os.path.join(tmp, "fa-negative-control.json")
            probe.write_receipt(control, {
                "kind": "fa_negative_control", "premise_holds": True,
                "llama_cpp_commit": "b96806d9",
                "llama_server_sha256": probe.sha256_bytes(b"BINARY-A"),
            })
            loaded = probe.load_r3_control(control, llamacpp_root=root)
            self.assertTrue(loaded["build_match"]["matches"])
            evidence = probe.flash_attn_evidence(
                probe.analyse_server_log(LOG_2026_COMPRESSED),
                probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"]),
                flags_source="own_launch_command", r3_control=loaded)
            self.assertEqual(evidence["r3_premise"], "verified")
            self.assertIn("flash-attention path ran", evidence["reading"])

    def test_a_control_left_behind_by_a_rebuild_stops_licensing_r3(self):
        # The concrete failure: build_llamacpp_cuda.sh runs again at a new commit, the old
        # control is still on disk, envelope_bench is given no --r3-control, and R3 keeps
        # being licensed for a binary the control never touched.
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_build(tmp, "1111aaaa", b"BINARY-B")   # rebuilt
            control = os.path.join(tmp, "fa-negative-control.json")
            probe.write_receipt(control, {
                "kind": "fa_negative_control", "premise_holds": True,
                "llama_cpp_commit": "b96806d9",
                "llama_server_sha256": probe.sha256_bytes(b"BINARY-A"),
            })
            loaded = probe.load_r3_control(control, llamacpp_root=root)
            self.assertFalse(loaded["build_match"]["matches"])
            evidence = probe.flash_attn_evidence(
                probe.analyse_server_log(LOG_2026_COMPRESSED),
                probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"]),
                flags_source="own_launch_command", r3_control=loaded)
            self.assertEqual(evidence["r3_premise"], "unverified_on_this_build")
            self.assertFalse(evidence["r3_usable"])
            self.assertNotIn("flash-attention path ran", evidence["reading"])
            self.assertTrue(any("DIFFERENT build" in item for item in evidence["basis"]))

    def test_a_control_that_cannot_be_joined_is_not_a_licence(self):
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED),
            probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"]),
            flags_source="docker_inspect", r3_control=R3_UNJOINED)
        self.assertEqual(evidence["r3_premise"], "unverified_on_this_build")
        self.assertFalse(evidence["r3_usable"])
        self.assertTrue(any("could NOT be joined" in item for item in evidence["basis"]))

    def test_a_void_control_from_another_build_does_not_void_this_one_either(self):
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED),
            probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"]),
            flags_source="own_launch_command",
            r3_control=dict(R3_VOID, build_match=BUILD_MISMATCHED))
        self.assertEqual(evidence["r3_premise"], "unverified_on_this_build")
        self.assertNotIn("R3 IS VOID", evidence["reading"])


class TestUnspecifiedFlagsAreNotObservations(unittest.TestCase):
    def test_forgetting_flags_source_does_not_make_flags_evidence(self):
        # Two probe receipts shipped with requested_flags_source null and an FA reading
        # nobody had licensed, because "unspecified" fell on the observed side of the line.
        flags = probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags, r3_control=R3_VERIFIED)
        self.assertEqual(evidence["flags_source"], "unspecified")
        self.assertFalse(evidence["flags_observed"])
        self.assertIsNone(evidence["requested_flash_attn"])
        self.assertFalse(evidence["r3_usable"])
        self.assertNotIn("flash-attention path ran", evidence["reading"])

    def test_an_observed_fa_off_is_read_as_evidence_too(self):
        # R1 used to be affirmative-only: an attached -fa off / f16 server reported "no
        # direct evidence either way" while its own basis line said -fa=off.
        flags = probe.parse_server_flags(["-fa", "off", "-ctk", "f16", "-ctv", "f16"])
        evidence = probe.flash_attn_evidence(
            probe.analyse_server_log(LOG_2026_COMPRESSED), flags,
            flags_source="docker_inspect", r3_control=R3_VERIFIED)
        self.assertIn("NOT requested", evidence["reading"])


class TestNegativeControlLicence(unittest.TestCase):
    """fa_negative_control.decide_premise: a failed NEG trial is not automatically a refusal."""

    def setUp(self):
        import fa_negative_control  # noqa: PLC0415
        self.mod = fa_negative_control

    def _trial(self, healthy, signals=(), exit_code=0, errors=()):
        return {"became_healthy": healthy, "refusal_signals": list(signals),
                "process_exit_code": exit_code, "error_lines": list(errors),
                "refusal_evidence_lines": list(signals) and ["a refusal line"]}

    def test_refusal_signals_found_in_the_real_2026_log(self):
        analysis = probe.analyse_server_log(LOG_REAL_REFUSAL_2026)
        signals, lines = self.mod.refusal_signals(analysis, LOG_REAL_REFUSAL_2026.splitlines())
        self.assertIn("kv_quant_refused", signals)
        self.assertTrue(any("quantized V cache requires flash_attn" in line for line in lines))

    def test_a_port_collision_is_not_a_refusal(self):
        bind_fail = (
            "0.00.101.000 E srv    main: server listen error: 127.0.0.1:18097 "
            "already in use\n"
            "0.00.101.900 E srv    main: exiting due to model loading error\n"
        )
        analysis = probe.analyse_server_log(bind_fail)
        signals, lines = self.mod.refusal_signals(analysis, bind_fail.splitlines())
        self.assertEqual(signals, [])
        self.assertEqual(lines, [])

    def test_an_oom_is_not_a_refusal(self):
        oom = ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 20480.00 MiB on "
               "device 2: cudaMalloc failed: out of memory\n"
               "llama_init_from_model: failed to initialize context\n")
        signals, _ = self.mod.refusal_signals(probe.analyse_server_log(oom), oom.splitlines())
        self.assertEqual(signals, [])

    def test_unrelated_neg_failure_is_inconclusive_not_a_licence(self):
        holds, observed = self.mod.decide_premise(
            self._trial(False, signals=(), exit_code=1, errors=["listen error: address in use"]),
            self._trial(True))
        self.assertIsNone(holds)
        self.assertIn("unrelated to KV quantization", observed)

    def test_a_real_refusal_licenses_the_premise(self):
        holds, observed = self.mod.decide_premise(
            self._trial(False, signals=("kv_quant_refused",), exit_code=1),
            self._trial(True))
        self.assertTrue(holds)
        self.assertIn("R3 LICENSED", observed)

    def test_neg_loading_anyway_voids_the_premise(self):
        holds, observed = self.mod.decide_premise(self._trial(True), self._trial(True))
        self.assertIs(holds, False)
        self.assertIn("R3 VOID", observed)

    def test_a_dead_positive_control_is_inconclusive(self):
        holds, observed = self.mod.decide_premise(
            self._trial(False, signals=("kv_quant_refused",)), self._trial(False))
        self.assertIsNone(holds)
        self.assertIn("positive control", observed)

    def test_port_guard_exists_and_refuses_an_occupied_port(self):
        import socket  # noqa: PLC0415

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            self.assertTrue(self.mod.port_is_answering(port))
            rc = self.mod.main([
                "--model", __file__, "--llamacpp-root", os.path.dirname(__file__),
                "--port", str(port), "--out-dir", tempfile.gettempdir(),
            ])
            # 3 = missing binary (checked first), 4 = port busy. Either way it does NOT run.
            self.assertIn(rc, (3, 4))
        finally:
            listener.close()


class TestEnvelopeServesWhatWeLaunched(unittest.TestCase):
    """A URL that answers is not proof that OUR server answered."""

    def setUp(self):
        import envelope_bench  # noqa: PLC0415
        self.mod = envelope_bench

    def test_dead_child_ends_the_wait_immediately(self):
        class DeadProcess(object):
            def poll(self):
                return 1

        handle = self.mod.ServerHandle()
        handle.mode = "host"
        handle.started_by_us = True
        handle.process = DeadProcess()

        class AlwaysHealthy(object):
            base_url = "http://127.0.0.1:18095"

            def health(self):
                return {"status": "ok"}   # a stranger on the same port

        result = self.mod.wait_healthy_watching_child(AlwaysHealthy(), handle, 30, poll_s=0.01)
        self.assertFalse(result["healthy"])
        self.assertEqual(result["child_exit_code"], 1)
        self.assertIn("exited rc=1", result["error"])

    def test_live_child_and_healthy_url_is_healthy(self):
        class LiveProcess(object):
            def poll(self):
                return None

        handle = self.mod.ServerHandle()
        handle.mode = "host"
        handle.started_by_us = True
        handle.process = LiveProcess()

        class Healthy(object):
            base_url = "http://127.0.0.1:18095"

            def health(self):
                return {"status": "ok"}

        result = self.mod.wait_healthy_watching_child(Healthy(), handle, 5, poll_s=0.01)
        self.assertTrue(result["healthy"])
        self.assertTrue(result["child_alive"])

    def test_attach_mode_has_no_child_to_watch(self):
        handle = self.mod.ServerHandle()
        handle.mode = "attach"

        class Healthy(object):
            base_url = "http://127.0.0.1:18080"

            def health(self):
                return {"status": "ok"}

        result = self.mod.wait_healthy_watching_child(Healthy(), handle, 5, poll_s=0.01)
        self.assertTrue(result["healthy"])
        self.assertIsNone(result["child_exit_code"])

    def test_served_model_mismatch_is_detected(self):
        check = self.mod.served_model_check(
            {"model_path": "/models/somebody-elses-27B.gguf", "n_ctx": 262144},
            262144, "/home/cereal/omni-campaign/export/gguf/Q35-08B-stock.Q4_K_M.gguf", 262144)
        self.assertFalse(check["served_model_matches_model_path"])
        self.assertTrue(check["reasons"])

    def test_served_model_match_and_ctx_clamp(self):
        check = self.mod.served_model_check(
            {"model_path": "/srv/gguf/Q35-08B-stock.Q4_K_M.gguf"}, 32768,
            "/home/cereal/omni-campaign/export/gguf/Q35-08B-stock.Q4_K_M.gguf", 262144)
        self.assertTrue(check["served_model_matches_model_path"])
        self.assertFalse(check["served_n_ctx_matches_requested"])
        self.assertTrue(any("n_ctx" in reason for reason in check["reasons"]))

    def test_no_props_model_path_is_unknown_not_a_match(self):
        check = self.mod.served_model_check({}, None, "/x/model.gguf", 4096)
        self.assertIsNone(check["served_model_matches_model_path"])


class TestTagSafetyEverywhere(unittest.TestCase):
    def test_shared_safe_tag_is_the_export_one(self):
        self.assertIs(export_mod.safe_tag, probe.safe_tag)

    def test_traversal_tag_is_rejected_by_every_entry_point(self):
        import envelope_bench  # noqa: PLC0415
        import fa_negative_control  # noqa: PLC0415
        import p40probe_check  # noqa: PLC0415

        for module in (envelope_bench, parity_mod, p40probe_check, fa_negative_control):
            with open(module.__file__, "r", encoding="utf-8") as source_handle:
                source = source_handle.read()
            self.assertIn("safe_tag(", source,
                          "%s interpolates --tag into a path without validating it"
                          % os.path.basename(module.__file__))

    def test_envelope_bench_rejects_a_traversal_tag(self):
        import envelope_bench  # noqa: PLC0415

        with self.assertRaises(ValueError):
            envelope_bench.main(["--tag", "../../escape", "--attach", "http://127.0.0.1:1"])


class TestExportReuseProvenance(unittest.TestCase):
    """A reused artifact must not be stamped with the current run's build hash."""

    def _tree(self, tmp, tag="T1", quants=("Q4_K_M",)):
        """A fake llamacpp root + hf dir + out dir where every artifact already exists.

        Because both the bf16 and the quant targets exist, main() takes the reuse path for
        all of them and never invokes a tool, so no real converter or quantizer is needed.
        """
        root = os.path.join(tmp, "llamacpp")
        os.makedirs(os.path.join(root, "bin"))
        os.makedirs(os.path.join(root, "llama.cpp"))
        with open(os.path.join(root, "llama.cpp", "convert_hf_to_gguf.py"), "w",
                  encoding="utf-8") as handle:
            handle.write("# fake converter, never executed on the reuse path\n")
        with open(os.path.join(root, "bin", "llama-quantize"), "w", encoding="utf-8") as handle:
            handle.write("# fake quantizer, never executed on the reuse path\n")
        hf_dir = os.path.join(tmp, "ckpt")
        os.makedirs(hf_dir)
        with open(os.path.join(hf_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"architectures": ["FakeForCausalLM"], "model_type": "fake",
                       "num_hidden_layers": 2, "hidden_size": 8, "vocab_size": 16,
                       "max_position_embeddings": 128}, handle)
        out_dir = os.path.join(tmp, "gguf")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "%s.bf16.gguf" % tag), "wb") as handle:
            handle.write(b"GGUF-BF16")
        for quant in quants:
            with open(os.path.join(out_dir, "%s.%s.gguf" % (tag, quant)), "wb") as handle:
                handle.write(b"GGUF-" + quant.encode())
        return root, hf_dir, out_dir

    def test_reuse_writes_a_quantize_stage_row(self):
        # The shipped receipt had two quant artifacts and ZERO quantize stages, so it could
        # not distinguish "quantized by this run" from "found on disk".
        with tempfile.TemporaryDirectory() as tmp:
            root, hf_dir, out_dir = self._tree(tmp)
            rc = export_mod.main(["--hf-dir", hf_dir, "--out-dir", out_dir, "--tag", "T1",
                                  "--llamacpp-root", root, "--quants", "Q4_K_M"])
            self.assertEqual(rc, 0)
            with open(os.path.join(out_dir, "export-receipt-T1.json"), encoding="utf-8") as h:
                receipt = json.load(h)
            stages = {row["stage"]: row for row in receipt["stages"]}
            self.assertIn("quantize:Q4_K_M", stages)
            self.assertEqual(stages["quantize:Q4_K_M"]["note"], "reused existing file")
            self.assertEqual(stages["convert"]["note"], "reused existing file")

    def test_a_reused_artifact_is_not_stamped_with_this_builds_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hf_dir, out_dir = self._tree(tmp)
            export_mod.main(["--hf-dir", hf_dir, "--out-dir", out_dir, "--tag", "T1",
                             "--llamacpp-root", root, "--quants", "Q4_K_M"])
            with open(os.path.join(out_dir, "export-receipt-T1.json"), encoding="utf-8") as h:
                receipt = json.load(h)
            for kind, info in receipt["artifacts"].items():
                self.assertIn(info["produced_by"], ("reused", "reused_unknown_provenance"),
                              "%s claimed to be produced by this run" % kind)
            self.assertFalse(receipt["artifacts_all_produced_by_this_run"])
            # No prior receipt existed, so provenance is honestly unknown rather than
            # inherited from the build this run happens to be pointed at.
            self.assertEqual(receipt["artifacts"]["Q4_K_M"]["produced_by"],
                             "reused_unknown_provenance")
            self.assertIsNone(receipt["artifacts"]["Q4_K_M"]["produced_by_build_receipt_sha256"])

    def test_provenance_is_carried_forward_from_the_prior_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hf_dir, out_dir = self._tree(tmp)
            export_mod.main(["--hf-dir", hf_dir, "--out-dir", out_dir, "--tag", "T1",
                             "--llamacpp-root", root, "--quants", "Q4_K_M"])
            path = os.path.join(out_dir, "export-receipt-T1.json")
            with open(path, encoding="utf-8") as handle:
                first = json.load(handle)
            # Pretend the first run produced them under a known build.
            for info in first["artifacts"].values():
                info["produced_by"] = "this_run"
                info["produced_by_build_receipt_sha256"] = "c" * 64
                info["produced_by_llama_cpp_commit"] = "b96806d9"
                info["produced_at"] = "2026-09-02T01:00:00Z"
            probe.write_receipt(path, first)
            export_mod.main(["--hf-dir", hf_dir, "--out-dir", out_dir, "--tag", "T1",
                             "--llamacpp-root", root, "--quants", "Q4_K_M"])
            with open(path, encoding="utf-8") as handle:
                second = json.load(handle)
            carried = second["artifacts"]["Q4_K_M"]
            self.assertEqual(carried["produced_by"], "reused")
            self.assertEqual(carried["produced_by_build_receipt_sha256"], "c" * 64)
            self.assertEqual(carried["produced_by_llama_cpp_commit"], "b96806d9")

    def test_same_tag_different_source_refuses_to_reuse(self):
        # Re-running with the same --tag but a different --hf-dir used to reuse the old
        # GGUFs and write a receipt whose source digest described a checkpoint the
        # artifacts did not come from.
        with tempfile.TemporaryDirectory() as tmp:
            root, hf_dir, out_dir = self._tree(tmp)
            export_mod.main(["--hf-dir", hf_dir, "--out-dir", out_dir, "--tag", "T1",
                             "--llamacpp-root", root, "--quants", "Q4_K_M"])
            other = os.path.join(tmp, "other-ckpt")
            os.makedirs(other)
            with open(os.path.join(other, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"architectures": ["OtherForCausalLM"], "model_type": "other",
                           "num_hidden_layers": 4, "hidden_size": 8, "vocab_size": 16,
                           "max_position_embeddings": 128}, handle)
            rc = export_mod.main(["--hf-dir", other, "--out-dir", out_dir, "--tag", "T1",
                                  "--llamacpp-root", root, "--quants", "Q4_K_M"])
            self.assertEqual(rc, 5)
            with open(os.path.join(out_dir, "export-receipt-T1.json"), encoding="utf-8") as h:
                receipt = json.load(h)
            self.assertIs(receipt["prior_receipt"]["source_matches_this_run"], False)
            self.assertFalse(receipt["reuse_allowed"])
            self.assertEqual(receipt["stages"][-1]["rc"], 5)

    def test_mismatched_reuse_can_be_forced_but_is_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hf_dir, out_dir = self._tree(tmp)
            export_mod.main(["--hf-dir", hf_dir, "--out-dir", out_dir, "--tag", "T1",
                             "--llamacpp-root", root, "--quants", "Q4_K_M"])
            other = os.path.join(tmp, "other-ckpt")
            os.makedirs(other)
            with open(os.path.join(other, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"architectures": ["OtherForCausalLM"], "model_type": "other",
                           "num_hidden_layers": 4, "hidden_size": 8, "vocab_size": 16,
                           "max_position_embeddings": 128}, handle)
            rc = export_mod.main(["--hf-dir", other, "--out-dir", out_dir, "--tag", "T1",
                                  "--llamacpp-root", root, "--quants", "Q4_K_M",
                                  "--allow-mismatched-reuse"])
            self.assertEqual(rc, 0)
            with open(os.path.join(out_dir, "export-receipt-T1.json"), encoding="utf-8") as h:
                receipt = json.load(h)
            self.assertTrue(receipt["reuse_allowed"])
            # The carried provenance is TRUE (that IS where the bytes came from) and it is
            # explicitly not this run's source.
            self.assertEqual(receipt["artifacts"]["Q4_K_M"]["produced_by"],
                             "reused_from_a_different_source")
            self.assertNotEqual(
                receipt["artifacts"]["Q4_K_M"]["produced_from_hf_dir"], receipt["hf_dir"])

    def test_prior_source_matches_helper(self):
        digest = {"directory_sha256": "d" * 64}
        self.assertEqual(export_mod.prior_source_matches(None, "/a", digest)[0], None)
        same = {"hf_dir": os.path.abspath("/a"), "source_directory_digest": digest}
        self.assertIs(export_mod.prior_source_matches(same, "/a", digest)[0], True)
        other = {"hf_dir": os.path.abspath("/b"), "source_directory_digest": digest}
        self.assertIs(export_mod.prior_source_matches(other, "/a", digest)[0], False)
        moved = {"hf_dir": os.path.abspath("/a"),
                 "source_directory_digest": {"directory_sha256": "e" * 64}}
        self.assertIs(export_mod.prior_source_matches(moved, "/a", digest)[0], False)


class TestR3ControlLoader(unittest.TestCase):
    def test_missing_control_returns_none_so_r3_stays_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(probe.load_r3_control(os.path.join(tmp, "nope.json")))

    def test_control_is_loaded_with_its_own_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fa-negative-control.json")
            digest = probe.write_receipt(path, {
                "kind": "fa_negative_control",
                "premise_holds": True,
                "llama_cpp_commit": "b96806d96061049a5b574269b049bf6241d63d46",
                "observed": "refuses with -fa off",
                "refusal_line": "V cache quantization requires flash_attn",
            })
            loaded = probe.load_r3_control(path)
            self.assertEqual(loaded["receipt_sha256"], digest)
            self.assertTrue(loaded["premise_holds"])
            self.assertEqual(loaded["path"], path)

    def test_candidate_names_include_both_spellings(self):
        import p40probe_check  # noqa: PLC0415

        self.assertIn("p40_256k", p40probe_check.CANDIDATES)
        self.assertIn("p40probe", p40probe_check.CANDIDATES)

    def test_resolve_container_separates_gone_from_stopped(self):
        """Three states: running, exited-but-present, and removed.

        Measured on this box 2026-09-02: `docker logs p40probe` answers
        `Error response from daemon: No such container: p40probe`, and so does `p40_256k`.
        Reporting that as "not running" would read like a container sitting there stopped
        with a log to inspect. There is no such container, and the receipt has to say the
        difference.
        """
        import p40probe_check  # noqa: PLC0415

        newline = chr(10)
        tab = chr(9)
        state = {"running": "", "all": ""}

        def fake_run(cmd, timeout=None):
            if cmd[:3] == ["docker", "ps", "-a"]:
                return 0, state["all"], ""
            if cmd[:2] == ["docker", "ps"]:
                return 0, state["running"], ""
            raise AssertionError("unexpected command %r" % (cmd,))

        original = p40probe_check.run
        p40probe_check.run = fake_run
        try:
            state["running"] = "p40_256k" + newline + "searxng" + newline
            state["all"] = ("p40_256k" + tab + "Up 3 hours" + newline
                            + "searxng" + tab + "Up 5 weeks" + newline)
            name, running, _all_running, exists, status = p40probe_check.resolve_container(None)
            self.assertEqual((name, running, exists), ("p40_256k", True, True))
            self.assertEqual(status, "Up 3 hours")

            # exited but present: still resolvable, and its log is still readable evidence
            state["running"] = "searxng" + newline
            state["all"] = ("p40_256k" + tab + "Exited (143) 2 hours ago" + newline
                            + "searxng" + tab + "Up 5 weeks" + newline)
            name, running, _all_running, exists, status = p40probe_check.resolve_container(None)
            self.assertEqual((name, running, exists), ("p40_256k", False, True))
            self.assertEqual(status, "Exited (143) 2 hours ago")

            # removed: there is nothing to read anywhere
            state["running"] = "searxng" + newline
            state["all"] = "searxng" + tab + "Up 5 weeks" + newline
            name, running, _all_running, exists, status = p40probe_check.resolve_container(None)
            self.assertEqual((name, running, exists), ("p40_256k", False, False))
            self.assertIsNone(status)

            # an explicit name is never silently swapped for a candidate that does exist
            state["running"] = "p40_256k" + newline
            state["all"] = "p40_256k" + tab + "Up 3 hours" + newline
            name, running, _all, exists, _status = p40probe_check.resolve_container("gpu0_256k")
            self.assertEqual((name, running, exists), ("gpu0_256k", False, False))
        finally:
            p40probe_check.run = original


class TestKvFitSearch(unittest.TestCase):
    """The bisect that turns "q8_0 does not fit at 256k" into "q8_0 fits up to N".

    Measured on this box 2026-09-02: a 27B Q4_K_M with q8_0/q8_0 KV at -c 262144 on a
    24 GB Tesla P40 loaded its weights (own peak 23,942 MiB) and then died in
    `graph_reserve: failed to allocate compute buffers`. The search logic is tested
    without a GPU here; the launching is exercised on the server.
    """

    def test_round_down_never_returns_a_partial_block(self):
        import kv_fit_search as fit  # noqa: PLC0415

        self.assertEqual(fit.round_down(262144, 4096), 262144)
        self.assertEqual(fit.round_down(262143, 4096), 258048)
        self.assertEqual(fit.round_down(100, 4096), 4096)

    def test_ceiling_that_loads_ends_the_search(self):
        import kv_fit_search as fit  # noqa: PLC0415

        self.assertIsNone(fit.next_context_to_try(262144, None, 8192, 262144, 4096))

    def test_bisect_walks_down_then_converges_to_the_resolution(self):
        import kv_fit_search as fit  # noqa: PLC0415

        # nothing fits yet: try the floor first, so a card that fits nothing costs 2 loads
        self.assertEqual(fit.next_context_to_try(None, 262144, 8192, 262144, 4096), 8192)
        # a fit and a failure: halve the interval
        self.assertEqual(fit.next_context_to_try(8192, 262144, 8192, 262144, 4096), 135168)
        # converged to one block: stop
        self.assertIsNone(fit.next_context_to_try(131072, 135168, 8192, 262144, 4096))

    def test_search_terminates_and_brackets_a_simulated_card(self):
        """Drive the whole bisect against a card whose true ceiling is known."""
        import kv_fit_search as fit  # noqa: PLC0415

        true_ceiling = 147456
        largest_fit = None
        smallest_fail = None
        ctx = 262144
        tried = []
        while ctx is not None and len(tried) < 12:
            tried.append(ctx)
            if ctx <= true_ceiling:
                largest_fit = ctx if largest_fit is None else max(largest_fit, ctx)
            else:
                smallest_fail = ctx if smallest_fail is None else min(smallest_fail, ctx)
            ctx = fit.next_context_to_try(largest_fit, smallest_fail, 8192, 262144, 4096)
        self.assertLessEqual(largest_fit, true_ceiling)
        self.assertGreater(smallest_fail, true_ceiling)
        self.assertLessEqual(smallest_fail - largest_fit, 4096)
        self.assertLessEqual(len(tried), 8, "bisect should not need more than 8 loads")

    def test_nothing_fits_is_reported_as_such(self):
        import kv_fit_search as fit  # noqa: PLC0415

        self.assertIsNone(fit.next_context_to_try(None, 8192, 8192, 262144, 4096))
        text = fit.summarise(None, 8192, "q8_0", 4096)
        self.assertIn("nothing in the searched range loaded", text)
        self.assertIn("every context tried loaded", fit.summarise(262144, None, "q4_0", 4096))
        bracket = fit.summarise(131072, 135168, "q8_0", 4096)
        self.assertIn("131072", bracket)
        self.assertIn("135168", bracket)

    def test_failure_lines_keeps_the_real_cuda_text(self):
        import kv_fit_search as fit  # noqa: PLC0415

        newline = chr(10)
        # verbatim from envelope/server-OC-Q36-27B-Q4KM-gpu3-q8kv.log on this box
        log = newline.join([
            "0.00.212.789 I srv    load_model: loading model ...",
            "1.06.706.993 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1360.28"
            " MiB on device 0: cudaMalloc failed: out of memory",
            "1.06.707.006 E graph_reserve: failed to allocate compute buffers",
            "1.06.719.902 E llama_init_from_model: failed to initialize the context:"
            " failed to allocate compute pp buffers",
        ])
        lines = fit.failure_lines(log)
        self.assertTrue(any("cudaMalloc failed: out of memory" in line for line in lines))
        self.assertTrue(any("failed to allocate compute buffers" in line for line in lines))
        self.assertFalse(any("loading model" in line for line in lines))
        self.assertEqual(fit.failure_lines(log, limit=1), lines[:1])

    def test_attempt_record_carries_its_own_footprint_and_timeout(self):
        """One attempt's record must stand alone: its own VRAM, its own timeout budget.

        The first real search summed eight sequential servers into one
        `own_peak_memory_used_mib` of 107,384 MiB -- a number that is no server's
        footprint. And its -c 262144 attempt waited 1,709 s against a requested
        --load-timeout of 420 with nothing in the receipt saying so.
        """
        import kv_fit_search as fit  # noqa: PLC0415

        class FakeProc(object):
            pid = 4242

            def poll(self):
                return None

        class FakeSampler(object):
            def __init__(self):
                # str keys, like the real GpuSampler (llama_probe.GpuSampler._poll_compute_apps
                # does `key = str(row["pid"])`). Keying this fake with an int 4242 exercised
                # try_context's first .get(pid) branch and would have passed even if the
                # .get(str(pid)) fallback that every real run depends on were deleted.
                self.peak_by_pid = {"4242": {"peak_memory_used_mib": 15815.0,
                                             "tracked_as_ours": True}}
                self.watched = []

            def watch_pid(self, pid):
                self.watched.append(pid)

        def fake_launch(handle, attempt_args, out_dir):
            handle.mode = "host"
            handle.process = FakeProc()
            handle.command = ["llama-server", "-c", str(attempt_args.ctx)]
            return handle

        def fake_wait(server, handle, deadline_s, poll_s=1.0):
            return {"healthy": False, "wait_s": 1709.62, "child_exit_code": None}

        args = argparse.Namespace(
            llamacpp_root="/x", model="/m.gguf", kv="q8_0", port=18096, fa="on",
            extra_args=None, bind="127.0.0.1", host_url="http://127.0.0.1",
            gpu_uuid=None, gpu_index=3, tag="T", load_timeout=420, settle_s=0.0,
        )
        original_launch, original_wait = fit.launch_host, fit.wait_healthy_watching_child
        fit.launch_host, fit.wait_healthy_watching_child = fake_launch, fake_wait
        try:
            sampler = FakeSampler()
            record = fit.try_context(args, 262144, sampler, tempfile.gettempdir())
        finally:
            fit.launch_host, fit.wait_healthy_watching_child = original_launch, original_wait

        self.assertEqual(record["ctx"], 262144)
        self.assertEqual(record["load_timeout_s"], 420)
        self.assertTrue(record["wait_exceeded_load_timeout"])
        self.assertEqual(record["own_peak_memory_used_mib"], 15815.0)
        self.assertEqual(sampler.watched, [4242])
        # nothing was started by us in this fake, so nothing may claim to be stopped
        self.assertFalse(record["teardown"]["stopped"])

    def test_module_stops_only_servers_it_started(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.assertEqual(
            docker_lifecycle_calls(os.path.join(here, "kv_fit_search.py")), [],
            "kv_fit_search.py must not touch container state",
        )


class TestRefusedPointSurvivesTheReceipt(unittest.TestCase):
    """A context point the server refuses must not take the whole receipt down.

    measure_context_point() returns early on a prefill error -- the fix that keeps one bad
    point from costing the points after it. But the early return used to omit the decode
    keys, and finish()'s summary indexed point["decode_tokens_per_s"] directly. The
    complete receipt was written at the top of finish(), the KeyError was raised at the
    bottom, main()'s `except` caught it, and finish() ran a SECOND time with partial=True
    and error="KeyError('decode_tokens_per_s')" -- overwriting a good sweep with a Python
    bug and leaving the .partial.json behind.
    """

    class RefusingServer(TestContextHeadroom.CeilingServer):
        """Tokenizes fine, then refuses every completion (a slot error at the top point)."""

        def completion(self, prompt, n_predict=64, **kwargs):
            self.refusals += 1
            raise urllib.error.HTTPError(
                "http://127.0.0.1:18090/completion", 400, "Bad Request", {}, None)

    class StubSampler(object):
        def stop(self):
            return {"peak_by_index": {}, "peak_by_pid": {},
                    "own_peak_memory_used_mib": None, "tracked_pids": []}

    def _refused_point(self):
        import envelope_bench  # noqa: PLC0415

        return envelope_bench.measure_context_point(
            self.RefusingServer(), 262144, "ZQ7-4418", 64, [0.5], do_needle=True,
            request_timeout=5.0, n_ctx=262144,
        )

    def test_a_refused_point_still_carries_the_decode_keys(self):
        point = self._refused_point()
        self.assertIn("HTTPError", point["error"])
        # present-and-null, not absent: every reader of a point may index these
        self.assertIn("decode_tokens_per_s", point)
        self.assertIsNone(point["decode_tokens_per_s"])
        self.assertIsNone(point["decode"])
        self.assertEqual(point["decode_tokens_requested"], 64)
        self.assertEqual(point["needle"], [])

    def test_finish_writes_a_complete_receipt_when_a_point_was_refused(self):
        import envelope_bench  # noqa: PLC0415

        point = self._refused_point()
        with tempfile.TemporaryDirectory() as tmp:
            receipt = {"kind": "envelope_bench", "tag": "T", "points_requested": 1,
                       "points": [point]}
            args = argparse.Namespace(out_dir=tmp, tag="T", keep_server=False)
            rc = envelope_bench.finish(receipt, args, envelope_bench.ServerHandle(),
                                       self.StubSampler(), time.time(), partial=False)
            self.assertEqual(rc, 0)
            with open(os.path.join(tmp, "envelope-T.json"), "r", encoding="utf-8") as fh:
                written = json.load(fh)
        # the sweep ran to the end, so it is NOT partial; the refusal is reported as data
        self.assertFalse(written["partial"])
        self.assertTrue(written["complete_sweep"])
        self.assertEqual(written["points_refused"], [262144])
        self.assertIn("HTTPError", written["points"][0]["error"])
        self.assertNotIn("KeyError", json.dumps(written))


class TestTruncatedSweepIsNotComplete(unittest.TestCase):
    """--max-wall-s cutting a sweep short must not produce `partial: false`.

    EXPORT_LANE_SPEC.md cites `partial: false` as the completeness marker. A sweep that
    stopped at 3 of 4 context points because the wall-clock budget ran out used to write
    exactly that, AND delete the .partial.json that would have shown the truncation --
    making a budget-killed run indistinguishable from a finished one.
    """

    class StubSampler(TestRefusedPointSurvivesTheReceipt.StubSampler):
        pass

    def test_the_helper_reads_stopped_early_as_partial(self):
        import envelope_bench  # noqa: PLC0415

        self.assertFalse(envelope_bench.sweep_is_partial({"points": []}))
        self.assertTrue(envelope_bench.sweep_is_partial(
            {"stopped_early": "max_wall_s reached before 262144-token point"}))
        self.assertTrue(envelope_bench.sweep_is_partial({"error": "boom"}))

    def test_a_deadline_break_keeps_the_partial_receipt_on_disk(self):
        import envelope_bench  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "envelope-T.partial.json")
            probe.write_receipt(partial_path, {"kind": "envelope_bench", "points": []})
            receipt = {
                "kind": "envelope_bench", "tag": "T", "points_requested": 4,
                "points": [{"target_tokens": 4096, "prefill_wall_s": 3.2,
                            "prefill_tokens_per_s": 1279.1, "decode_tokens_per_s": 38.9,
                            "needle": []}],
                "stopped_early": "max_wall_s reached before 262144-token point",
            }
            args = argparse.Namespace(out_dir=tmp, tag="T", keep_server=False)
            rc = envelope_bench.finish(
                receipt, args, envelope_bench.ServerHandle(), self.StubSampler(),
                time.time(), partial=envelope_bench.sweep_is_partial(receipt))
            self.assertEqual(rc, 1)
            with open(os.path.join(tmp, "envelope-T.json"), "r", encoding="utf-8") as fh:
                written = json.load(fh)
            # the intermediate receipt that shows the truncation is NOT deleted
            self.assertTrue(os.path.exists(partial_path))
        self.assertTrue(written["partial"])
        self.assertFalse(written["complete_sweep"])
        self.assertEqual(written["points_measured"], 1)
        self.assertEqual(written["points_requested"], 4)

    def test_main_marks_a_budget_truncated_sweep_partial_end_to_end(self):
        """Drive main() in --attach mode with a 1 s budget: the loop breaks mid-sweep."""
        import envelope_bench  # noqa: PLC0415

        class FakeSampler(object):
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return self

            def watch_pid(self, pid):
                pass

            def stop(self):
                return {"peak_by_index": {}, "peak_by_pid": {},
                        "own_peak_memory_used_mib": None, "tracked_pids": []}

        class SlowServer(TestContextHeadroom.CeilingServer):
            """A CeilingServer that also answers /props and burns budget per request."""

            def props(self):
                return {"model_path": None, "n_ctx": self.n_ctx, "total_slots": 1}

            def completion(self, prompt, n_predict=64, **kwargs):
                time.sleep(0.6)
                return super().completion(prompt, n_predict=n_predict, **kwargs)

        server = SlowServer(n_ctx=8192)
        originals = (envelope_bench.GpuSampler, envelope_bench.LlamaServer,
                     envelope_bench.wait_healthy_watching_child, envelope_bench.host_facts)
        envelope_bench.GpuSampler = FakeSampler
        envelope_bench.LlamaServer = lambda *args, **kwargs: server
        envelope_bench.wait_healthy_watching_child = (
            lambda *args, **kwargs: {"healthy": True, "wait_s": 0.1})
        envelope_bench.host_facts = lambda: {"hostname": "test"}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc = envelope_bench.main([
                    "--tag", "TRUNC", "--out-dir", tmp,
                    "--attach", "http://127.0.0.1:1", "--ctx", "8192",
                    "--prefill-points", "4096,8192", "--needle-points", "8192",
                    "--needle-depths", "0.5", "--max-wall-s", "1",
                ])
                self.assertTrue(os.path.exists(
                    os.path.join(tmp, "envelope-TRUNC.partial.json")),
                    "the truncation evidence was deleted")
                with open(os.path.join(tmp, "envelope-TRUNC.json"), "r",
                          encoding="utf-8") as fh:
                    written = json.load(fh)
        finally:
            (envelope_bench.GpuSampler, envelope_bench.LlamaServer,
             envelope_bench.wait_healthy_watching_child,
             envelope_bench.host_facts) = originals
        self.assertEqual(rc, 1)
        self.assertTrue(written["partial"])
        self.assertFalse(written["complete_sweep"])
        self.assertIn("max_wall_s", written["stopped_early"])
        self.assertLess(written["points_measured"], written["points_requested"])


class TestControlWithoutABinaryHashIsNotALicence(unittest.TestCase):
    """A control that never recorded which binary it measured licenses nothing.

    The control this lane shipped first (sha256 38e80ce8...) had no llama_server_sha256.
    Joined on commit alone it matched ANY binary built from that commit -- including the
    separate imatrix build that is deferred precisely because it would change the binary.
    The join now requires the llama-server hash on both sides.
    """

    def _fake_build(self, tmp, commit, server_bytes):
        return TestR3BuildJoin._fake_build(self, tmp, commit, server_bytes)

    def test_a_control_with_no_binary_hash_does_not_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_build(tmp, "b96806d9", b"BINARY-A")
            control = os.path.join(tmp, "fa-negative-control.json")
            probe.write_receipt(control, {
                "kind": "fa_negative_control", "premise_holds": True,
                "llama_cpp_commit": "b96806d9",   # commit only -- the shipped shape
            })
            loaded = probe.load_r3_control(control, llamacpp_root=root)
            join = loaded["build_match"]
            self.assertIsNone(join["matches"], "commit alone must not be a join")
            self.assertFalse(join["required_field_matched"])
            self.assertEqual(join["fields_missing_on_control"], ["llama_server_sha256"])
            evidence = probe.flash_attn_evidence(
                probe.analyse_server_log(LOG_2026_COMPRESSED),
                probe.parse_server_flags(["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"]),
                flags_source="own_launch_command", r3_control=loaded)
            self.assertEqual(evidence["r3_premise"], "unverified_on_this_build")
            self.assertFalse(evidence["r3_usable"])
            self.assertNotIn("flash-attention path ran", evidence["reading"])

    def test_the_rebuild_at_the_same_commit_is_the_concrete_failure(self):
        """Same commit, different binary: the old join answered `matches: True`."""
        with tempfile.TemporaryDirectory() as tmp:
            # the imatrix build: same commit, different --arch, different bytes
            root = self._fake_build(tmp, "b96806d9", b"BINARY-IMATRIX")
            control = os.path.join(tmp, "fa-negative-control.json")
            probe.write_receipt(control, {
                "kind": "fa_negative_control", "premise_holds": True,
                "llama_cpp_commit": "b96806d9",
            })
            loaded = probe.load_r3_control(control, llamacpp_root=root)
            self.assertIsNot(loaded["build_match"]["matches"], True)
            self.assertTrue(any("does not record which binary" in reason
                                for reason in loaded["build_match"]["reasons"]))

    def test_a_matching_binary_hash_is_still_a_licence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_build(tmp, "b96806d9", b"BINARY-A")
            control = os.path.join(tmp, "fa-negative-control.json")
            probe.write_receipt(control, {
                "kind": "fa_negative_control", "premise_holds": True,
                "llama_cpp_commit": "b96806d9",
                "llama_server_sha256": probe.sha256_bytes(b"BINARY-A"),
            })
            join = probe.load_r3_control(control, llamacpp_root=root)["build_match"]
            self.assertTrue(join["matches"])
            self.assertTrue(join["required_field_matched"])
            self.assertEqual(join["fields_compared"],
                             ["llama_cpp_commit", "llama_server_sha256"])

    def test_the_binary_hash_alone_joins_when_the_commit_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "llamacpp")
            os.makedirs(os.path.join(root, "bin"))
            binary = os.path.join(root, "bin", "llama-server")
            with open(binary, "wb") as fh:
                fh.write(b"BINARY-A")            # no build-receipt.json at all
            control = os.path.join(tmp, "fa-negative-control.json")
            probe.write_receipt(control, {
                "kind": "fa_negative_control", "premise_holds": True,
                "llama_server_sha256": probe.sha256_bytes(b"BINARY-A"),
            })
            join = probe.load_r3_control(control, llamacpp_root=root)["build_match"]
            self.assertTrue(join["matches"])
            self.assertEqual(join["fields_compared"], ["llama_server_sha256"])

    def test_the_control_shipped_in_this_lane_is_joinable_to_its_binary(self):
        """The on-disk control must carry the hash, or it cannot license anything."""
        # local checkout keeps receipts under receipts/; the server copy is beside the tool
        here = os.path.dirname(os.path.abspath(__file__))
        found = [path for path in (
            os.path.join(here, "receipts", "fa-control", "fa-negative-control.json"),
            os.path.join(here, "fa-control", "fa-negative-control.json"),
        ) if os.path.exists(path)]
        if not found:
            self.skipTest("no shipped control in this checkout")
        for shipped in found:
            with open(shipped, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertTrue(payload.get("llama_server_sha256"),
                            "%s records no llama_server_sha256, so it cannot be joined to "
                            "any binary and licenses nothing" % shipped)


class TestBorrowedHfColumnIsLabelled(unittest.TestCase):
    """--reference copies another receipt's HF answers; the receipt must say so.

    Without provenance the borrowed column reads exactly like a measured one, and if the
    reference receipt is re-run in place (write_receipt os.replace's it) the borrowing
    receipt silently describes bytes that no longer exist.
    """

    def _reference(self, tmp, chat=False, bank_sha256=None):
        path = os.path.join(tmp, "parity-REF.json")
        probe.write_receipt(path, {
            "kind": "gguf_hf_parity", "tag": "REF", "created_at": "2026-09-01T00:00:00Z",
            "bank_id": "export_parity_v1", "bank_sha256": bank_sha256,
            "hf_dir": "/home/cereal/models/Qwen3.5-0.8B", "hf_dtype": "bfloat16",
            "hf_loader": "AutoModelForCausalLM", "chat_template_applied": chat,
            "rows": [{"id": "PB1-01", "hf": {"content": " 1081"}}],
        })
        return path

    def _run(self, tmp, reference_kwargs=None):
        import parity_check  # noqa: PLC0415

        here = os.path.dirname(os.path.abspath(__file__))
        argv = ["--tag", "BORROWED", "--out-dir", tmp, "--skip-gguf", "--skip-hf",
                "--bank", os.path.join(here, "parity_bank.json"),
                "--reference", self._reference(tmp, **(reference_kwargs or {}))]
        rc = parity_check.main(argv)
        with open(os.path.join(tmp, "parity-BORROWED.json"), "r", encoding="utf-8") as fh:
            return rc, json.load(fh)

    def test_the_receipt_names_the_bytes_the_hf_column_came_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, written = self._run(tmp)
            self.assertEqual(rc, 0)
            reference = written["hf_reference"]
            self.assertEqual(written["hf_side_source"], "reference")
            self.assertEqual(reference["sha256"],
                             probe.sha256_file(os.path.join(tmp, "parity-REF.json")))
            self.assertEqual(reference["hf_dir"], "/home/cereal/models/Qwen3.5-0.8B")
            self.assertEqual(reference["hf_dtype"], "bfloat16")
            self.assertEqual(reference["tag"], "REF")
            self.assertGreaterEqual(reference["rows_borrowed"], 1)

    def test_every_borrowed_row_is_marked_as_borrowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, written = self._run(tmp)
            borrowed = [row for row in written["rows"] if row["hf"]]
            self.assertTrue(borrowed)
            for row in borrowed:
                self.assertEqual(row["hf_source"], "reference")
                self.assertEqual(len(row["hf_reference_sha256"]), 64)
            # rows the reference has nothing for stay empty AND unmarked
            for row in written["rows"]:
                if not row["hf"]:
                    self.assertIsNone(row["hf_source"])

    def test_a_reference_rendered_differently_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, written = self._run(tmp, reference_kwargs={"chat": True})
            self.assertFalse(written["hf_reference"]["chat_rendering_matches_this_run"])
            # the stub reference carries no bank_sha256 either
            self.assertFalse(written["hf_reference"]["bank_matches_this_run"])

    def test_the_human_read_says_the_hf_side_was_borrowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            with open(os.path.join(tmp, "READ-BORROWED.txt"), "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("HF SIDE: BORROWED", text)
            self.assertIn("Nothing in the HF column was measured in this run", text)
            self.assertIn("HF (transformers, reference)", text)


class TestPeakSeriesConsistency(unittest.TestCase):
    """own_peak > whole-card peak is impossible at one instant, so it must be flagged.

    envelope-OC-Q36-27B-Q4KM-gpu3-q8kv.json reports own_peak_memory_used_mib 23,942 MiB on
    a card whose whole-card peak is 15,245 MiB in the same 32-sample run with no sampler
    errors, and the spec quotes the 23,942 figure as fact. One series missed the ramp.
    """

    def _sampler(self, own_mib, card_mib):
        sampler = probe.GpuSampler(interval=0.01, indices=[3])
        sampler.peak = {3: {"peak_memory_used_mib": card_mib, "memory_total_mib": 24576.0}}
        sampler.peak_by_pid = {"1234": {"peak_memory_used_mib": own_mib,
                                        "tracked_as_ours": True, "gpu_index": 3}}
        sampler.track_pids = {1234}
        sampler._poll_once = lambda: None   # no nvidia-smi on the workstation
        return sampler

    def test_the_impossible_pair_is_flagged(self):
        payload = self._sampler(23942.0, 15245.0).stop()
        self.assertTrue(payload["own_peak_exceeds_whole_card_peak"])
        self.assertEqual(payload["whole_card_peak_max_mib"], 15245.0)
        self.assertIn("EXCEEDS", payload["peak_series_note"])
        # reported, never repaired: both numbers survive as sampled
        self.assertEqual(payload["own_peak_memory_used_mib"], 23942.0)

    def test_a_consistent_pair_is_not_flagged(self):
        payload = self._sampler(21522.0, 24121.0).stop()
        self.assertFalse(payload["own_peak_exceeds_whole_card_peak"])
        self.assertIsNone(payload["peak_series_note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
