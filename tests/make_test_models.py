"""Small models to exercise dynamic input and custom class parsing."""
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
prediction = np.array([[[100, 200], [100, 200], [50, 50], [50, 50], [.9, .1], [.1, .8]]], dtype=np.float32)
node = helper.make_node("Constant", [], ["predictions"], value=numpy_helper.from_array(prediction))
graph = helper.make_graph([node], "two-classes", [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 3, "h", "w"])], [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 6, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, folder/"two_classes.onnx")
(folder/"classes.txt").write_text("first\nsecond\n", encoding="utf-8")
