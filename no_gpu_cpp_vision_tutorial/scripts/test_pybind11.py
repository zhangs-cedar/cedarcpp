#!/usr/bin/env python3
import ng_py_infer

engine = ng_py_infer.Engine("fake_model.bin")
print("model_path:", engine.model_path)
print(engine.infer_vector([0.1, 0.2, 0.3]))
print(engine.infer_vector([0.8, 0.9, 0.7]))
