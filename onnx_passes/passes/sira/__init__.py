from typing import Any

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator

from onnx_passes.ops import link_ops
from onnx_passes.ops.sira import DOMAIN as SIRA_DOMAIN
from onnx_passes.passes.base import Transformation

# Operators whose SIRA handler has a 1-to-1 signature (all inputs -> 4 outputs each)
_DIRECT_OPS = {
    "Abs",
    "Add",
    "BatchNormalization",
    "BipolarQuant",
    "Ceil",
    "Div",
    "Exp",
    "Flatten",
    "FloatQuant",
    "Floor",
    "Gather",
    "Identity",
    "Log",
    "MatMul",
    "Mul",
    "Neg",
    "Pow",
    "Quant",
    "Relu",
    "Reshape",
    "Round",
    "Shape",
    "Sigmoid",
    "Sqrt",
    "Sub",
    "Tanh",
    "Transpose",
    "Trunc",
}


_ROUNDING_MODES = {
    "ROUND": 0,
    "CEIL": 1,
    "FLOOR": 2,
    "UP": 3,
    "DOWN": 4,
    "HALF_UP": 5,
    "HALF_DOWN": 6,
}


def _attr_ints(node: ir.Node, name: str, default: list[int]) -> list[int]:
    if (attr := node.attributes.get(name)) is not None:
        return list(attr.as_ints())
    return list(default)


def _attr_int(node: ir.Node, name: str, default: int) -> int:
    if (attr := node.attributes.get(name)) is not None:
        return int(attr.as_int())
    return int(default)


def _attr_float(node: ir.Node, name: str, default: float) -> float:
    if (attr := node.attributes.get(name)) is not None:
        return float(attr.as_float())
    return float(default)


def _attr_str(node: ir.Node, name: str, default: str) -> str:
    if (attr := node.attributes.get(name)) is not None:
        return str(attr.as_string())
    return str(default)


def _make_const(graph: ir.Graph, array: np.ndarray, name: str) -> ir.Value:
    val = ir.Value(name=name, const_value=ir.tensor(array))
    graph.initializers[val.name] = val
    return val


def _infer_conv_pool_shape(
    x_shape: tuple[int, ...],
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
    dilations: list[int],
    out_channels: int,
    ceil_mode: int = 0,
) -> ir.Shape:
    srank = len(kernel_shape)
    out_dims = [x_shape[0], out_channels]
    for i in range(srank):
        in_d = x_shape[2 + i]
        k = dilations[i] * (kernel_shape[i] - 1) + 1
        p = pads[i] + pads[i + srank]
        if ceil_mode:
            d = int(np.ceil((in_d + p - k) / strides[i])) + 1
        else:
            d = (in_d + p - k) // strides[i] + 1
        out_dims.append(d)
    return ir.Shape(out_dims)


def _infer_matmul_shape(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> ir.Shape | None:
    if len(a_shape) == 0 or len(b_shape) == 0:
        return None
    if len(a_shape) == 1 and len(b_shape) == 1:
        return ir.Shape([])
    if len(a_shape) == 1:
        return ir.Shape(list(b_shape[:-2]) + [b_shape[-1]])
    if len(b_shape) == 1:
        return ir.Shape(list(a_shape[:-1]))
    try:
        batch = list(np.broadcast_shapes(a_shape[:-2], b_shape[:-2]))
    except ValueError:
        return None
    return ir.Shape(batch + [a_shape[-2], b_shape[-1]])


def _tag_tuple(
    orig_name: str,
    tup: tuple[ir.Value, ir.Value, ir.Value, ir.Value],
):
    for val, role in zip(tup, ("min", "max", "scale", "bias")):
        val.meta["sira_origin"] = (orig_name, role)


class RangeAnalysis_v1(Transformation):
    """Transform an ONNX model into its SIRA scaled-integer range analysis graph.

    Each tensor ``v`` in the original graph is represented by four tensors
    ``(v_min, v_max, v_scale, v_bias)``. Each operator is replaced by its
    corresponding range analysis handler in ``onnx_passes.ops.sira``.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        graph.sort()

        # Target opset >= 18 so all SIRA ONNX functions use modern ONNX ops
        orig_opset = graph.opset_imports.get("", 18)
        graph.opset_imports[""] = max(orig_opset, 18)
        graph.opset_imports[SIRA_DOMAIN] = 1

        # Shared scalar NaN constant for non-scaled-integer scale/bias
        nan_const = _make_const(
            graph, np.array(np.nan, dtype=np.float32), "__sira_nan"
        )

        # Map original ir.Value -> (v_min, v_max, v_scale, v_bias)
        range_map: dict[
            ir.Value, tuple[ir.Value, ir.Value, ir.Value, ir.Value]
        ] = {}
        # Track statically known integer shape tensors (Shape/Gather/Unsqueeze/Concat)
        const_shapes: dict[ir.Value, np.ndarray] = {}

        # Map original initializers
        for init_val in list(graph.initializers.values()):
            range_map[init_val] = (init_val, init_val, nan_const, nan_const)

        # Replace graph inputs: initializers listed in inputs become constants;
        # dynamic inputs expand to (min, max, scale, bias)
        new_inputs: list[ir.Value] = []
        for inp in list(graph.inputs):
            if inp.const_value is not None or inp in range_map:
                range_map[inp] = (inp, inp, nan_const, nan_const)
                continue

            inp_min = ir.Value(
                name=f"{inp.name}__sira_min",
                shape=inp.shape,
                type=inp.type,
            )
            inp_max = ir.Value(
                name=f"{inp.name}__sira_max",
                shape=inp.shape,
                type=inp.type,
            )
            bcast_shape = (
                ir.Shape([None] * len(inp.shape))
                if inp.shape is not None
                else ir.Shape([])
            )
            inp_scale = ir.Value(
                name=f"{inp.name}__sira_scale",
                shape=bcast_shape,
                type=ir.TensorType(ir.DataType.FLOAT),
            )
            inp_bias = ir.Value(
                name=f"{inp.name}__sira_bias",
                shape=bcast_shape,
                type=ir.TensorType(ir.DataType.FLOAT),
            )
            tup = (inp_min, inp_max, inp_scale, inp_bias)
            _tag_tuple(inp.name, tup)
            range_map[inp] = tup
            new_inputs.extend(tup)

        graph.inputs.clear()
        graph.inputs.extend(new_inputs)

        new_nodes: list[ir.Node] = []
        old_nodes = list(graph)
        counter = 0

        def next_name(prefix: str) -> str:
            nonlocal counter
            counter += 1
            return f"__sira_{prefix}_{counter}"

        def emit_sira_node(
            op_type: str,
            inputs: list[ir.Value],
            attributes: dict[str, ir.Attr] | None = None,
            tag_name: str | None = None,
            out_shape: ir.Shape | None = None,
            out_type: Any = None,
        ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
            base = tag_name or next_name(op_type)
            outs = tuple(
                ir.Value(name=f"{base}__sira_{role}")
                for role in ("min", "max", "scale", "bias")
            )
            if out_shape is not None:
                outs[0].shape = out_shape
                outs[1].shape = out_shape
                outs[0].type = out_type or ir.TensorType(ir.DataType.FLOAT)
                outs[1].type = out_type or ir.TensorType(ir.DataType.FLOAT)
            elif out_type is not None:
                outs[0].type = out_type
                outs[1].type = out_type
            if tag_name is not None:
                _tag_tuple(tag_name, outs)
            node = ir.Node(
                domain=SIRA_DOMAIN,
                op_type=op_type,
                inputs=inputs,
                outputs=list(outs),
                attributes=attributes or {},
                name=next_name(op_type),
            )
            new_nodes.append(node)
            return outs  # type: ignore[return-value]

        def const_tuple(arr: np.ndarray, prefix: str) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
            c = _make_const(graph, arr, next_name(prefix))
            return (c, c, nan_const, nan_const)

        for node in old_nodes:
            op_type = node.op_type

            if op_type == "Constant":
                new_nodes.append(node)
                c = node.outputs[0]
                range_map[c] = (c, c, nan_const, nan_const)
                continue

            if len(node.outputs) != 1:
                raise NotImplementedError(
                    f"SIRA RangeAnalysis does not support multi-output op: {node.domain}:{op_type}"
                )

            out_val = node.outputs[0]
            out_name = out_val.name or next_name(op_type)

            def get_in(idx: int) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
                return range_map[node.inputs[idx]]

            def flat_inputs(indices: list[int] | None = None) -> list[ir.Value]:
                idxs = indices if indices is not None else list(range(len(node.inputs)))
                res: list[ir.Value] = []
                for i in idxs:
                    if node.inputs[i] is not None:
                        res.extend(get_in(i))
                return res

            if op_type == "Conv":
                x_val = node.inputs[0]
                w_val = node.inputs[1]
                # Determine spatial rank from weights or kernel_shape
                w_shape = w_val.shape or range_map[w_val][0].shape
                if (ks_attr := node.attributes.get("kernel_shape")) is not None:
                    ks = list(ks_attr.as_ints())
                elif w_shape is not None and len(w_shape) > 2:
                    ks = [int(d) for d in w_shape[2:]]
                else:
                    ks = [1, 1]
                srank = len(ks)
                group = _attr_int(node, "group", 1)
                is_dw = (
                    group > 1
                    and w_shape is not None
                    and len(w_shape) >= 2
                    and w_shape[0] == group
                    and w_shape[1] == 1
                )
                conv_op = "DepthwiseConv" if is_dw else "Conv"
                attrs = {
                    "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, ks),
                    "dilations": ir.Attr("dilations", ir.AttributeType.INTS, _attr_ints(node, "dilations", [1] * srank)),
                    "strides": ir.Attr("strides", ir.AttributeType.INTS, _attr_ints(node, "strides", [1] * srank)),
                    "pads": ir.Attr("pads", ir.AttributeType.INTS, _attr_ints(node, "pads", [0] * (2 * srank))),
                    "group": ir.Attr("group", ir.AttributeType.INT, group),
                    "spatial_axes": ir.Attr("spatial_axes", ir.AttributeType.INTS, list(range(2, 2 + srank))),
                    "weight_axes": ir.Attr("weight_axes", ir.AttributeType.INTS, list(range(1, 2 + srank))),
                }
                has_bias = len(node.inputs) > 2 and node.inputs[2] is not None
                conv_out_shape = out_val.shape
                if conv_out_shape is None:
                    xs = range_map[x_val][0].shape
                    ws = range_map[w_val][0].shape
                    if xs is not None and ws is not None and xs.is_static() and ws.is_static():
                        conv_out_shape = _infer_conv_pool_shape(
                            tuple(xs.dims),
                            ks,
                            attrs["strides"].as_ints(),
                            attrs["pads"].as_ints(),
                            attrs["dilations"].as_ints(),
                            int(ws.dims[0]),
                        )
                conv_out = emit_sira_node(
                    conv_op,
                    flat_inputs([0, 1]),
                    attrs,
                    tag_name=None if has_bias else out_name,
                    out_shape=conv_out_shape,
                    out_type=None if has_bias else out_val.type,
                )
                if has_bias:
                    b_shape = np.array([1, -1] + [1] * srank, dtype=np.int64)
                    b_reshaped = emit_sira_node(
                        "Reshape",
                        [*get_in(2), *const_tuple(b_shape, "conv_b_shape")],
                    )
                    res = emit_sira_node(
                        "Add",
                        [*conv_out, *b_reshaped],
                        tag_name=out_name,
                        out_shape=conv_out_shape,
                        out_type=out_val.type,
                    )
                else:
                    res = conv_out
                range_map[out_val] = res

            elif op_type == "Gemm":
                alpha = _attr_float(node, "alpha", 1.0)
                beta = _attr_float(node, "beta", 1.0)
                trans_a = _attr_int(node, "transA", 0)
                trans_b = _attr_int(node, "transB", 0)
                a_in = get_in(0)
                b_in = get_in(1)
                if trans_a != 0:
                    a_in = emit_sira_node(
                        "Transpose",
                        list(a_in),
                        {"perm": ir.Attr("perm", ir.AttributeType.INTS, [1, 0])},
                    )
                if trans_b != 0:
                    b_in = emit_sira_node(
                        "Transpose",
                        list(b_in),
                        {"perm": ir.Attr("perm", ir.AttributeType.INTS, [1, 0])},
                    )
                has_c = len(node.inputs) > 2 and node.inputs[2] is not None
                mm_out = emit_sira_node(
                    "MatMul",
                    [*a_in, *b_in],
                    tag_name=None if (has_c or alpha != 1.0) else out_name,
                    out_shape=None if (has_c or alpha != 1.0) else out_val.shape,
                    out_type=None if (has_c or alpha != 1.0) else out_val.type,
                )
                if alpha != 1.0:
                    alpha_t = np.array(alpha, dtype=np.float32)
                    mm_out = emit_sira_node(
                        "Mul",
                        [*mm_out, *const_tuple(alpha_t, "gemm_alpha")],
                        tag_name=None if has_c else out_name,
                        out_shape=None if has_c else out_val.shape,
                        out_type=None if has_c else out_val.type,
                    )
                if has_c:
                    c_in = get_in(2)
                    if beta != 1.0:
                        beta_t = np.array(beta, dtype=np.float32)
                        c_in = emit_sira_node("Mul", [*c_in, *const_tuple(beta_t, "gemm_beta")])
                    res = emit_sira_node(
                        "Add",
                        [*mm_out, *c_in],
                        tag_name=out_name,
                        out_shape=out_val.shape,
                        out_type=out_val.type,
                    )
                else:
                    res = mm_out
                range_map[out_val] = res

            elif op_type == "MaxPool":
                ks = _attr_ints(node, "kernel_shape", [1, 1])
                srank = len(ks)
                attrs = {
                    "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, ks),
                    "strides": ir.Attr("strides", ir.AttributeType.INTS, _attr_ints(node, "strides", [1] * srank)),
                    "pads": ir.Attr("pads", ir.AttributeType.INTS, _attr_ints(node, "pads", [0] * (2 * srank))),
                    "dilations": ir.Attr("dilations", ir.AttributeType.INTS, _attr_ints(node, "dilations", [1] * srank)),
                    "ceil_mode": ir.Attr("ceil_mode", ir.AttributeType.INT, _attr_int(node, "ceil_mode", 0)),
                    "storage_order": ir.Attr("storage_order", ir.AttributeType.INT, _attr_int(node, "storage_order", 0)),
                }
                mp_shape = out_val.shape
                if mp_shape is None:
                    xs = range_map[node.inputs[0]][0].shape
                    if xs is not None and xs.is_static():
                        mp_shape = _infer_conv_pool_shape(
                            tuple(xs.dims), ks, attrs["strides"].as_ints(),
                            attrs["pads"].as_ints(), attrs["dilations"].as_ints(),
                            int(xs.dims[1]), attrs["ceil_mode"].as_int(),
                        )
                range_map[out_val] = emit_sira_node(
                    "MaxPool", flat_inputs([0]), attrs, tag_name=out_name,
                    out_shape=mp_shape, out_type=out_val.type,
                )

            elif op_type == "AveragePool":
                ks = _attr_ints(node, "kernel_shape", [1, 1])
                srank = len(ks)
                attrs = {
                    "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, ks),
                    "strides": ir.Attr("strides", ir.AttributeType.INTS, _attr_ints(node, "strides", [1] * srank)),
                    "pads": ir.Attr("pads", ir.AttributeType.INTS, _attr_ints(node, "pads", [0] * (2 * srank))),
                    "ceil_mode": ir.Attr("ceil_mode", ir.AttributeType.INT, _attr_int(node, "ceil_mode", 0)),
                    "count_include_pad": ir.Attr("count_include_pad", ir.AttributeType.INT, _attr_int(node, "count_include_pad", 0)),
                    "spatial_axes": ir.Attr("spatial_axes", ir.AttributeType.INTS, list(range(2, 2 + srank))),
                }
                ap_shape = out_val.shape
                if ap_shape is None:
                    xs = range_map[node.inputs[0]][0].shape
                    if xs is not None and xs.is_static():
                        ap_shape = _infer_conv_pool_shape(
                            tuple(xs.dims), ks, attrs["strides"].as_ints(),
                            attrs["pads"].as_ints(), [1] * srank,
                            int(xs.dims[1]), attrs["ceil_mode"].as_int(),
                        )
                range_map[out_val] = emit_sira_node(
                    "AveragePool", flat_inputs([0]), attrs, tag_name=out_name,
                    out_shape=ap_shape, out_type=out_val.type,
                )

            elif op_type == "Pad":
                mode = _attr_str(node, "mode", "constant")
                attrs = {"mode": ir.Attr("mode", ir.AttributeType.STRING, mode)}
                if len(node.inputs) >= 2 and node.inputs[1] is not None:
                    pads_in = get_in(1)
                    if len(node.inputs) >= 3 and node.inputs[2] is not None:
                        val_in = get_in(2)
                    else:
                        val_in = const_tuple(np.array(0.0, dtype=np.float32), "pad_val")
                else:
                    pads_arr = np.array(_attr_ints(node, "pads", [0]), dtype=np.int64)
                    val_arr = np.array(_attr_float(node, "value", 0.0), dtype=np.float32)
                    pads_in = const_tuple(pads_arr, "pads")
                    val_in = const_tuple(val_arr, "pad_val")
                pad_shape = out_val.shape
                if pad_shape is None:
                    xs = range_map[node.inputs[0]][0].shape
                    if xs is not None and xs.is_static():
                        if len(node.inputs) >= 2 and node.inputs[1] is not None:
                            pc = ir.convenience.get_const_tensor(node.inputs[1])
                            p_list = [int(x) for x in pc.numpy().flatten()] if pc is not None else None
                        else:
                            p_list = _attr_ints(node, "pads", [0] * (2 * len(xs.dims)))
                        if p_list is not None and len(p_list) == 2 * len(xs.dims):
                            r = len(xs.dims)
                            pad_shape = ir.Shape([int(xs.dims[i]) + p_list[i] + p_list[i + r] for i in range(r)])
                range_map[out_val] = emit_sira_node(
                    "Pad", [*get_in(0), *pads_in, *val_in], attrs, tag_name=out_name,
                    out_shape=pad_shape, out_type=out_val.type,
                )

            elif op_type == "Clip":
                if len(node.inputs) >= 2:
                    lo_in = get_in(1) if node.inputs[1] is not None else const_tuple(np.array(-np.inf, dtype=np.float32), "clip_min")
                    hi_in = get_in(2) if len(node.inputs) >= 3 and node.inputs[2] is not None else const_tuple(np.array(np.inf, dtype=np.float32), "clip_max")
                else:
                    lo_in = const_tuple(np.array(_attr_float(node, "min", -np.inf), dtype=np.float32), "clip_min")
                    hi_in = const_tuple(np.array(_attr_float(node, "max", np.inf), dtype=np.float32), "clip_max")
                range_map[out_val] = emit_sira_node(
                    "Clip", [*get_in(0), *lo_in, *hi_in], {}, tag_name=out_name
                )

            elif op_type == "Unsqueeze":
                if len(node.inputs) >= 2 and node.inputs[1] is not None:
                    axes_c = ir.convenience.get_const_tensor(node.inputs[1])
                    axes = [int(x) for x in axes_c.numpy().flatten()] if axes_c is not None else [0]
                else:
                    axes = _attr_ints(node, "axes", [0])
                attrs = {"axes": ir.Attr("axes", ir.AttributeType.INTS, axes)}
                if node.inputs[0] in const_shapes:
                    arr = const_shapes[node.inputs[0]]
                    for ax in sorted(axes):
                        arr = np.expand_dims(arr, axis=ax)
                    const_shapes[out_val] = arr
                range_map[out_val] = emit_sira_node(
                    "Unsqueeze", flat_inputs([0]), attrs, tag_name=out_name,
                    out_shape=ir.Shape(const_shapes[out_val].shape) if out_val in const_shapes else out_val.shape,
                    out_type=out_val.type or (ir.TensorType(ir.DataType.INT64) if out_val in const_shapes else None),
                )

            elif op_type == "Concat":
                axis = _attr_int(node, "axis", 0)
                attrs = {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)}
                c_parts = []
                for inp in node.inputs:
                    if inp in const_shapes:
                        c_parts.append(const_shapes[inp])
                    elif (tc := ir.convenience.get_const_tensor(inp)) is not None:
                        c_parts.append(tc.numpy())
                    else:
                        c_parts = []
                        break
                if c_parts:
                    const_shapes[out_val] = np.concatenate(c_parts, axis=axis)
                c_shape = ir.Shape(const_shapes[out_val].shape) if out_val in const_shapes else out_val.shape
                c_type = out_val.type or (ir.TensorType(ir.DataType.INT64) if out_val in const_shapes else None)
                if len(node.inputs) == 1:
                    range_map[out_val] = emit_sira_node(
                        "Identity", flat_inputs([0]), {}, tag_name=out_name,
                        out_shape=c_shape, out_type=c_type,
                    )
                else:
                    acc = get_in(0)
                    for i in range(1, len(node.inputs)):
                        is_last = i == len(node.inputs) - 1
                        acc = emit_sira_node(
                            "Concat2",
                            [*acc, *get_in(i)],
                            attrs,
                            tag_name=out_name if is_last else None,
                            out_shape=c_shape if is_last else None,
                            out_type=c_type if is_last else None,
                        )
                    range_map[out_val] = acc

            elif op_type == "Softmax":
                default_ax = 1 if orig_opset < 13 else -1
                axis = _attr_int(node, "axis", default_ax)
                attrs = {"axes": ir.Attr("axes", ir.AttributeType.INTS, [axis])}
                range_map[out_val] = emit_sira_node("Softmax", flat_inputs([0]), attrs, tag_name=out_name)

            elif op_type == "Transpose":
                s0 = range_map[node.inputs[0]][0].shape
                if (perm_attr := node.attributes.get("perm")) is not None:
                    perm = list(perm_attr.as_ints())
                elif s0 is not None:
                    perm = list(reversed(range(len(s0))))
                else:
                    perm = [1, 0]
                attrs = {"perm": ir.Attr("perm", ir.AttributeType.INTS, perm)}
                t_shape = out_val.shape
                if t_shape is None and s0 is not None and s0.is_static():
                    t_shape = ir.Shape([s0.dims[p] for p in perm])
                range_map[out_val] = emit_sira_node(
                    "Transpose", flat_inputs([0]), attrs, tag_name=out_name,
                    out_shape=t_shape, out_type=out_val.type,
                )

            elif op_type in _DIRECT_OPS:
                # Filter attributes to those supported by the SIRA handler
                attrs = dict(node.attributes)
                if op_type == "Quant":
                    rm_str = _attr_str(node, "rounding_mode", "ROUND").upper()
                    attrs = {
                        "signed": ir.Attr("signed", ir.AttributeType.INT, _attr_int(node, "signed", 1)),
                        "narrow": ir.Attr("narrow", ir.AttributeType.INT, _attr_int(node, "narrow", 0)),
                        "rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.INT, _ROUNDING_MODES.get(rm_str, 0)),
                    }
                elif op_type == "BipolarQuant":
                    attrs = {}
                elif op_type == "Trunc":
                    rm_str = _attr_str(node, "rounding_mode", "FLOOR").upper()
                    attrs = {
                        "rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.INT, _ROUNDING_MODES.get(rm_str, 2)),
                    }
                elif op_type == "FloatQuant":
                    attrs = {
                        "has_inf": ir.Attr("has_inf", ir.AttributeType.INT, _attr_int(node, "has_inf", 0)),
                        "has_nan": ir.Attr("has_nan", ir.AttributeType.INT, _attr_int(node, "has_nan", 0)),
                        "has_subnormal": ir.Attr("has_subnormal", ir.AttributeType.INT, _attr_int(node, "has_subnormal", 1)),
                        "rounding_mode": ir.Attr("rounding_mode", ir.AttributeType.STRING, _attr_str(node, "rounding_mode", "round")),
                        "saturation": ir.Attr("saturation", ir.AttributeType.INT, _attr_int(node, "saturation", 1)),
                    }
                elif op_type == "BatchNormalization":
                    attrs = {
                        "epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, _attr_float(node, "epsilon", 1e-5)),
                        "momentum": ir.Attr("momentum", ir.AttributeType.FLOAT, _attr_float(node, "momentum", 0.9)),
                    }
                elif op_type == "Shape":
                    attrs = {}
                    s0 = range_map[node.inputs[0]][0].shape
                    if s0 is not None and s0.is_static():
                        const_shapes[out_val] = np.array([int(d) for d in s0.dims], dtype=np.int64)
                elif op_type == "Gather":
                    if node.inputs[0] in const_shapes and (ic := ir.convenience.get_const_tensor(node.inputs[1])) is not None:
                        ax = _attr_int(node, "axis", 0)
                        const_shapes[out_val] = np.take(const_shapes[node.inputs[0]], ic.numpy(), axis=ax)

                inferred_shape = out_val.shape
                if inferred_shape is None:
                    s0 = range_map[node.inputs[0]][0].shape
                    if op_type in {"Quant", "BipolarQuant", "Trunc", "FloatQuant", "Relu", "BatchNormalization", "Ceil", "Floor", "Round", "Log", "Exp", "Sqrt", "Sigmoid", "Tanh", "Abs", "Neg", "Identity", "Cast"}:
                        inferred_shape = s0
                    elif op_type in {"Add", "Sub", "Mul", "Div", "Pow"}:
                        s1 = range_map[node.inputs[1]][0].shape
                        if s0 is not None and s1 is not None and s0.is_static() and s1.is_static():
                            try:
                                inferred_shape = ir.Shape(np.broadcast_shapes(tuple(s0.dims), tuple(s1.dims)))
                            except ValueError:
                                inferred_shape = None
                    elif op_type == "MatMul":
                        s1 = range_map[node.inputs[1]][0].shape
                        if s0 is not None and s1 is not None and s0.is_static() and s1.is_static():
                            inferred_shape = _infer_matmul_shape(tuple(s0.dims), tuple(s1.dims))
                    elif op_type == "Transpose" and s0 is not None and s0.is_static():
                        perm = attrs["perm"].as_ints() if "perm" in attrs else list(reversed(range(len(s0.dims))))
                        inferred_shape = ir.Shape([s0.dims[p] for p in perm])
                    elif op_type == "Flatten" and s0 is not None and s0.is_static():
                        ax = _attr_int(node, "axis", 1)
                        d0 = int(np.prod(s0.dims[:ax])) if ax > 0 else 1
                        d1 = int(np.prod(s0.dims[ax:]))
                        inferred_shape = ir.Shape([d0, d1])
                    elif op_type == "Reshape" and s0 is not None and s0.is_static():
                        tc = ir.convenience.get_const_tensor(node.inputs[1])
                        t_arr = tc.numpy() if tc is not None else const_shapes.get(node.inputs[1])
                        if t_arr is not None:
                            t = [int(x) for x in t_arr.flatten()]
                            total = int(np.prod(s0.dims))
                            known = int(np.prod([x for x in t if x != -1]))
                            inferred_shape = ir.Shape([total // known if x == -1 else x for x in t])
                    elif op_type == "Shape" and out_val in const_shapes:
                        inferred_shape = ir.Shape(const_shapes[out_val].shape)
                    elif op_type == "Gather" and out_val in const_shapes:
                        inferred_shape = ir.Shape(const_shapes[out_val].shape)
                out_t = out_val.type
                if out_t is None and out_val in const_shapes:
                    out_t = ir.TensorType(ir.DataType.INT64)
                range_map[out_val] = emit_sira_node(
                    op_type,
                    flat_inputs(),
                    attrs,
                    tag_name=out_name,
                    out_shape=inferred_shape,
                    out_type=out_t,
                )

            else:
                raise NotImplementedError(
                    f"No SIRA range analysis handler for operator: {node.domain}:{op_type}"
                )

        # Replace nodes in graph
        graph.remove(old_nodes, safe=False)
        graph.extend(new_nodes)

        # Update graph outputs to (y_min, y_max, y_scale, y_bias)
        # Broadcast output scale and bias to y_min's shape so output shapes are
        # statically known and valid for ONNX checker / ONNX Runtime.
        new_outputs: list[ir.Value] = []
        for out in list(graph.outputs):
            o_min, o_max, o_scale, o_bias = range_map[out]
            out_type = out.type or ir.TensorType(ir.DataType.FLOAT)
            out_shape = out.shape or ir.Shape([])
            o_min.type = o_min.type or out_type
            o_min.shape = o_min.shape or out_shape
            o_max.type = o_max.type or out_type
            o_max.shape = o_max.shape or out_shape

            zero_out = ir.Value(name=next_name("out_zero"))
            graph.append(ir.Node("", "Sub", inputs=[o_min, o_min], outputs=[zero_out]))

            o_scale_b = ir.Value(name=f"{out.name}__sira_out_scale", type=out_type, shape=out_shape)
            o_bias_b = ir.Value(name=f"{out.name}__sira_out_bias", type=out_type, shape=out_shape)
            graph.append(ir.Node("", "Add", inputs=[zero_out, o_scale], outputs=[o_scale_b]))
            graph.append(ir.Node("", "Add", inputs=[zero_out, o_bias], outputs=[o_bias_b]))
            new_outputs.extend((o_min, o_max, o_scale_b, o_bias_b))
        graph.outputs.clear()
        graph.outputs.extend(new_outputs)

        # Record all range tuples in model metadata so intermediate names can be
        # resolved even after canonical renaming
        model.meta["sira_range_map"] = {
            orig_val.name: tup for orig_val, tup in range_map.items() if orig_val.name
        }

        return ir.passes.PassResult(model, True)


def _real_range_from_tuple(
    v_min: np.ndarray,
    v_max: np.ndarray,
    v_scale: np.ndarray,
    v_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute real-valued (min, max) from a (min, max, scale, bias) tuple."""
    if not (np.isnan(v_scale).any() or np.isnan(v_bias).any()):
        a = v_min * v_scale + v_bias
        b = v_max * v_scale + v_bias
        return np.minimum(a, b), np.maximum(a, b)
    return v_min, v_max


def compute_accumulator_bitwidth(z_min: np.ndarray, z_max: np.ndarray) -> int:
    """Compute minimum two's complement accumulator bitwidth (SIRA Section 4.2)."""
    max_abs = max(int(np.max(np.abs(z_min))), int(np.max(np.abs(z_max))) + 1)
    if max_abs <= 1:
        return 2
    return int(np.ceil(np.log2(max_abs))) + 1


def evaluate_ranges(
    range_model: ir.Model,
    input_ranges: dict[str, tuple[np.ndarray, np.ndarray]] | list[tuple[np.ndarray, np.ndarray]],
    input_scales: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    backend: str = "onnxruntime",
    intermediates: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Execute a SIRA range analysis model and return ranges per original tensor.

    Parameters
    ----------
    range_model : ir.Model
        Model transformed by :class:`RangeAnalysis_v1`.
    input_ranges : dict or list
        Mapping from original input name (or positional list) to ``(min, max)``.
    input_scales : dict, optional
        Optional mapping from original input name to ``(scale, bias)``. Defaults
        to ``(nan, nan)`` (non-scaled-integer inputs).
    backend : str
        Execution backend: ``"onnxruntime"`` or ``"reference"``.
    intermediates : bool
        If True, return ranges for all intermediate tensors as well as outputs.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Mapping from original tensor name to ``(min, max, scale, bias)``.
    """
    model = range_model.clone(deep_copy=True) if intermediates and backend == "onnxruntime" else range_model
    link_ops(model)

    # Build feed dict
    feed: dict[str, np.ndarray] = {}
    nan_arr = np.array(np.nan, dtype=np.float32)

    def _nan_for(inp_val: ir.Value) -> np.ndarray:
        rank = len(inp_val.shape) if inp_val.shape is not None else 0
        return np.full((1,) * rank, np.nan, dtype=np.float32)

    if isinstance(input_ranges, list):
        # Positional: every 4 inputs correspond to 1 original dynamic input
        for idx, (r_min, r_max) in enumerate(input_ranges):
            inp_min = model.graph.inputs[4 * idx]
            inp_max = model.graph.inputs[4 * idx + 1]
            inp_s = model.graph.inputs[4 * idx + 2]
            inp_b = model.graph.inputs[4 * idx + 3]
            dtype = inp_min.dtype.numpy() if inp_min.dtype else np.float32
            feed[inp_min.name] = np.asarray(r_min, dtype=dtype)
            feed[inp_max.name] = np.asarray(r_max, dtype=dtype)
            feed[inp_s.name] = _nan_for(inp_s)
            feed[inp_b.name] = _nan_for(inp_b)
    else:
        for i in range(0, len(model.graph.inputs), 4):
            inp_min, inp_max, inp_s, inp_b = model.graph.inputs[i : i + 4]
            orig_name = inp_min.meta.get("sira_origin", (inp_min.name.removesuffix("__sira_min"), "min"))[0]
            r_min, r_max = input_ranges[orig_name]
            dtype = inp_min.dtype.numpy() if inp_min.dtype else np.float32
            feed[inp_min.name] = np.asarray(r_min, dtype=dtype)
            feed[inp_max.name] = np.asarray(r_max, dtype=dtype)
            if input_scales and orig_name in input_scales:
                s, b = input_scales[orig_name]
                rank = len(inp_s.shape) if inp_s.shape is not None else 0
                s_arr = np.asarray(s, dtype=np.float32)
                b_arr = np.asarray(b, dtype=np.float32)
                if s_arr.ndim < rank:
                    s_arr = s_arr.reshape((1,) * (rank - s_arr.ndim) + s_arr.shape)
                if b_arr.ndim < rank:
                    b_arr = b_arr.reshape((1,) * (rank - b_arr.ndim) + b_arr.shape)
                feed[inp_s.name] = s_arr
                feed[inp_b.name] = b_arr
            else:
                feed[inp_s.name] = _nan_for(inp_s)
                feed[inp_b.name] = _nan_for(inp_b)

    # Collect tagged values in graph
    tagged: dict[str, list[ir.Value | None]] = {}
    for val in list(model.graph.inputs) + [o for n in model.graph for o in n.outputs]:
        if (origin := val.meta.get("sira_origin")) is not None:
            orig_name, role = origin
            slot = {"min": 0, "max": 1, "scale": 2, "bias": 3}[role]
            tagged.setdefault(orig_name, [None, None, None, None])[slot] = val

    if backend == "onnxruntime":
        if intermediates:
            existing = set(model.graph.outputs)
            for vals in tagged.values():
                for v in vals:
                    if v is not None and v not in existing:
                        if v.type is None:
                            v.type = ir.TensorType(ir.DataType.FLOAT)
                        if v.shape is None:
                            v.shape = ir.Shape([])
                        model.graph.outputs.append(v)
                        existing.add(v)
        proto = ir.to_proto(model)
        sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
        out_names = [o.name for o in model.graph.outputs]
        raw_out = sess.run(out_names, feed)
        ctx = dict(zip(out_names, raw_out))
    else:
        proto = ir.to_proto(model)
        sess = ReferenceEvaluator(proto)
        ctx = sess.run(None, feed, intermediate=intermediates)

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if tagged:
        for orig_name, vals in tagged.items():
            if all(v is not None and v.name in ctx for v in vals):
                results[orig_name] = tuple(np.asarray(ctx[v.name]) for v in vals)  # type: ignore[assignment]
    else:
        for i in range(0, len(model.graph.outputs), 4):
            o_min, o_max, o_s, o_b = model.graph.outputs[i : i + 4]
            results[o_min.name] = (
                np.asarray(ctx[o_min.name]),
                np.asarray(ctx[o_max.name]),
                np.asarray(ctx[o_s.name]),
                np.asarray(ctx[o_b.name]),
            )

    return results
