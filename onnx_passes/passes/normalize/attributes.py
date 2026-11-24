from onnx_passes.passes.base import Transformation
from onnx_passes.passes.verify import Verify

import onnx_ir as ir

# Common passes already implemented in ONNX IR
import onnx_ir.passes.common


def _strip_default_training_mode(model: ir.Model) -> None:
    """Drop BatchNormalization training_mode=0 as ONNX IR cleanup does.

    ONNX IR's RemoveUnusedNodesPass (part of the cleanup after each pass) pops
    the inference-mode training_mode attribute from BatchNormalization when
    removing unused optional outputs. Adding it back as a default attribute
    would oscillate forever in exhaustive sequences, e.g., canonicalize.
    """
    graphs = [model.graph, *model.functions.values()]
    for graph in graphs:
        for node in ir.traversal.RecursiveGraphIterator(graph):
            if node.op_type != "BatchNormalization" or node.domain not in {"", "ai.onnx"}:
                continue
            attr = node.attributes.get("training_mode")
            if (
                attr is not None and not attr.is_ref()
                and attr.type == ir.AttributeType.INT and attr.as_int() == 0
            ):
                node.attributes.pop("training_mode")


def _attribute_snapshot(model: ir.Model) -> list:
    graphs = [model.graph, *model.functions.values()]
    return [
        (id(node), tuple(sorted(node.attributes)))
        for graph in graphs
        for node in ir.traversal.RecursiveGraphIterator(graph)
    ]


class AddDefaultAttributes_v1(Transformation, Verify):
    """Add default values for optional attributes not present in nodes."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        before = _attribute_snapshot(model)
        ir.passes.common.AddDefaultAttributesPass()(model)
        _strip_default_training_mode(model)
        return ir.passes.PassResult(
            model, modified=_attribute_snapshot(model) != before
        )
