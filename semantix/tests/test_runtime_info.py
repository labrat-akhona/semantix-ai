"""Tests for the shipped-file list and runtime snapshot in quantized_nli."""

from __future__ import annotations

import pytest

from semantix.judges import quantized_nli as qnli


@pytest.mark.parametrize(
    ("machine", "cpuinfo"),
    [
        ("x86_64", "flags : fpu avx2"),
        ("x86_64", "flags : fpu avx2 avx512f"),
        ("x86_64", "flags : fpu avx2 avx512f avx512_vnni"),
        ("aarch64", ""),
        ("AMD64", ""),  # Windows has no /proc/cpuinfo
    ],
)
def test_auto_selected_file_is_a_shipped_file(monkeypatch, machine, cpuinfo):
    monkeypatch.setattr(qnli.platform, "machine", lambda: machine)
    monkeypatch.setattr(qnli, "_read_cpuinfo", lambda: cpuinfo)
    assert qnli._detect_onnx_variant() in qnli.ONNX_VARIANTS


def test_onnx_variants_lists_four_distinct_files():
    assert len(qnli.ONNX_VARIANTS) == 4
    assert len(set(qnli.ONNX_VARIANTS)) == 4
    assert all(v.startswith("onnx/") and v.endswith(".onnx") for v in qnli.ONNX_VARIANTS)


def test_runtime_info_reports_relevant_cpu_flags(monkeypatch):
    monkeypatch.setattr(qnli.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(qnli, "_read_cpuinfo", lambda: "flags : fpu sse avx2 avx512f avx512_vnni")
    info = qnli.runtime_info()
    assert set(info) == {"onnxruntime", "python", "os", "machine", "cpu_flags", "auto_variant"}
    assert info["cpu_flags"] == "avx2 avx512f avx512_vnni"
    assert info["auto_variant"] == "onnx/model_qint8_avx512_vnni.onnx"


def test_runtime_info_without_proc_cpuinfo(monkeypatch):
    monkeypatch.setattr(qnli.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(qnli, "_read_cpuinfo", lambda: "")
    info = qnli.runtime_info()
    assert info["cpu_flags"] == "unavailable (no /proc/cpuinfo)"
    assert info["auto_variant"] == "onnx/model_quint8_avx2.onnx"
