"""Tests for the onnx-passes command-line interface."""

from __future__ import annotations

import pydantic_settings
import pytest

from onnx_passes.cli import main as cli
from onnx_passes.passes.base import list_passes


def test_list_passes_discovers_unimported_modules():
    identifiers = {f"{m}.{n}_v{v}" for m, n, v, _ in list_passes()}
    assert "onnx_passes.passes.legalize.UpgradeOpset_v1" in identifiers
    assert "onnx_passes.passes.streamline.StreamlineScales_v1" in identifiers
    # Ad-hoc sequences (e.g., the CLI's own Passes_v0) are not listed
    assert all(i.startswith("onnx_passes.passes.") for i in identifiers)


def test_list_passes_filters_case_insensitive():
    names = {n for _, n, _, _ in list_passes(["LEGALIZE", "StreamlineThr"])}
    assert names == {"NormalizeDomains", "UpgradeOpset", "StreamlineThresholds"}
    assert list_passes(["no-such-pass"]) == []


def test_cli_list_passes(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Do not pick up a passes.yaml
    pydantic_settings.CliApp.run(
        cli.OnnxPasses, cli_args=["--list-passes", "legalize"]
    )
    out = capsys.readouterr().out
    assert "UpgradeOpset_v1" in out
    assert "2 passes" in out


def test_cli_requires_model_without_list_passes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        pydantic_settings.CliApp.run(cli.OnnxPasses, cli_args=[])
    assert error.value.code == 2
