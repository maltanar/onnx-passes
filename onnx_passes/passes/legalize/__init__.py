"""Legalize models from external exporters for further transformation.

Models exported by older tools (e.g., Brevitas or FINN) often do not satisfy
the assumptions of the remaining passes:

* Custom operators are placed in legacy domains which have since been renamed,
  e.g., ``finn.custom_op.general`` is now ``qonnx.custom_op.general``.
* Custom-op domains used by nodes are not declared as opset imports, which
  breaks ONNX shape inference and checking.
* The default ONNX opset is too old for the operator functions linked by
  onnx-passes, e.g., ``Constant(value_float=...)`` requires opset 11.

All passes in this module are no-ops on models which are already legal and can
thus be applied unconditionally in front of any pass sequence.
"""

import onnx_ir as ir
import onnxscript.version_converter

from onnx_passes.passes.base import Transformation

# Legacy custom-op domains mapped to their current names
LEGACY_DOMAINS = {"finn.custom_op.general": "qonnx.custom_op.general"}

# Minimum default ONNX opset version required by the onnx-passes operator
# functions and rewrite rules
MIN_DEFAULT_OPSET = 19


def normalize_domains(model: ir.Model) -> bool:
    """Renames legacy custom-op domains and declares missing opset imports.

    Returns whether the model has been modified.
    """
    modified = False
    opset_imports = model.graph.opset_imports

    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        if node.domain in LEGACY_DOMAINS:
            node.domain = LEGACY_DOMAINS[node.domain]
            modified = True

        # Undeclared custom-op domains default to version 1, the only version
        # of all custom-op domains understood by onnx-passes so far
        if node.domain not in opset_imports and node.domain != "ai.onnx":
            opset_imports[node.domain] = 1
            modified = True

    # Carry over the version of renamed domains which are declared under their
    # legacy name, dropping the now unused legacy import
    for legacy, current in LEGACY_DOMAINS.items():
        if legacy in opset_imports:
            opset_imports.setdefault(current, opset_imports[legacy])
            del opset_imports[legacy]
            modified = True

    return modified


def upgrade_opset(model: ir.Model, target: int = MIN_DEFAULT_OPSET) -> bool:
    """Upgrades the default ONNX opset to at least the target version.

    Models already at or above the target version are left unchanged, i.e.,
    this never downgrades. Returns whether the model has been modified.
    """
    current = model.graph.opset_imports.get(
        "", model.graph.opset_imports.get("ai.onnx")
    )

    if current is None or current >= target:
        return False

    # The version converter inlines all model functions before conversion,
    # which would expand linked custom-op functions: Detach these and restore
    # them afterward, functions carry their own opset imports
    functions = dict(model.functions)
    model.functions.clear()

    # Fallback to the ONNX C API as the onnxscript version converter does not
    # cover upgrades from older opsets
    onnxscript.version_converter.convert_version(model, target, fallback=True)

    model.functions.update(functions)

    # The version converter silently leaves the model unchanged if the C API
    # fails, which should be reported as the following passes would fail
    if model.graph.opset_imports.get("") != target:
        raise RuntimeError(
            f"Failed to upgrade default ONNX opset from {current} to {target}"
        )

    return True


class NormalizeDomains_v1(Transformation):
    """Rename legacy custom-op domains and declare missing opset imports."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, normalize_domains(model))

    @staticmethod
    def cleanup(result: ir.passes.PassResult):
        # Skip the default cleanup: Linking custom-op functions requires a
        # sufficiently recent default opset, which is only ensured by the
        # following UpgradeOpset pass. Nothing to clean up after renaming.
        return result


class UpgradeOpset_v1(Transformation):
    """Upgrade the default ONNX opset to the minimum required version."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return ir.passes.PassResult(model, upgrade_opset(model))
