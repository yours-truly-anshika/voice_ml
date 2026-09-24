# scripts/verify_tflite_models.py
"""Verify FP32 and INT8 TFLite models.

Requirements:
- TensorFlow (tf.lite.Interpreter)
- NumPy
- The repository contains:
  * models/tflite/dscnn_1.0s_fp32.tflite
  * models/tflite/dscnn_1.0s_int8.tflite
  * dataset/features_1.0s/dataset_features.npz (with X_test)
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf

# Paths (relative to repo root)
FP32_MODEL_PATH = os.path.join("models", "tflite", "dscnn_1.0s_fp32.tflite")
INT8_MODEL_PATH = os.path.join("models", "tflite", "dscnn_1.0s_int8.tflite")
DATASET_PATH = os.path.join("dataset", "features_1.0s", "dataset_features.npz")

def load_tflite_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Model file not found: {path}")
        sys.exit(1)
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"[ERROR] Failed to load TFLite model {path}: {e}")
        sys.exit(1)

def print_tensor_info(interpreter, tag="Input"):
    details = interpreter.get_input_details() if tag.lower() == "input" else interpreter.get_output_details()
    d = details[0]
    shape = d["shape"].tolist()
    dtype = d["dtype"]
    quant = d.get("quantization", (0, 0))
    print(f"{tag} tensor:")
    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    if quant[0] != 0 or quant[1] != 0:
        print(f"  Quantization: scale={quant[0]}, zero_point={quant[1]}")
    else:
        print("  Quantization: None (float model)")

def check_fully_integer(interpreter):
    in_dtype = interpreter.get_input_details()[0]["dtype"]
    out_dtype = interpreter.get_output_details()[0]["dtype"]
    if in_dtype not in (np.int8, np.uint8):
        return False, "Input dtype not integer"
    if out_dtype not in (np.int8, np.uint8):
        return False, "Output dtype not integer"
    if hasattr(interpreter, "get_tensor_details"):
        for t in interpreter.get_tensor_details():
            if t["dtype"] == tf.float32:
                return False, f"Found float tensor: {t['name']}"
    return True, "All tensors integer"

def run_inference(interpreter, inputs):
    input_details = interpreter.get_input_details()[0]
    dtype = input_details["dtype"]
    quant = input_details.get("quantization", (0, 0))
    if dtype == np.float32:
        input_tensor = inputs.astype(np.float32)
    else:
        scale, zero_point = quant
        if scale == 0:
            scale = 1.0
        input_tensor = ((inputs / scale) + zero_point).astype(dtype)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    out_details = interpreter.get_output_details()[0]
    out = interpreter.get_tensor(out_details["index"])
    if out_details["dtype"] != np.float32:
        scale, zero_point = out_details.get("quantization", (0, 0))
        if scale != 0:
            out = (out.astype(np.float32) - zero_point) * scale
    return out.squeeze()

def main():
    fp32_interp = load_tflite_model(FP32_MODEL_PATH)
    int8_interp = load_tflite_model(INT8_MODEL_PATH)
    print("=== Model tensor info ===")
    print("FP32 model:")
    print_tensor_info(fp32_interp, "Input")
    print_tensor_info(fp32_interp, "Output")
    print("INT8 model:")
    print_tensor_info(int8_interp, "Input")
    print_tensor_info(int8_interp, "Output")
    ok, msg = check_fully_integer(int8_interp)
    print("INT8 fully integer verification:")
    print(f"  {msg}")
    if not ok:
        sys.exit(1)
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)
    npz = np.load(DATASET_PATH)
    X_test = npz["X_test"]
    sample = X_test[:5]
    fp32_outputs = []
    int8_outputs = []
    for i in range(sample.shape[0]):
        inp = sample[i]
        if inp.ndim == 3:
            inp = np.expand_dims(inp, axis=0)
        fp32_outputs.append(run_inference(fp32_interp, inp))
        int8_outputs.append(run_inference(int8_interp, inp))
    fp32_outputs = np.stack(fp32_outputs)
    int8_outputs = np.stack(int8_outputs)
    abs_diff = np.abs(fp32_outputs - int8_outputs)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)
    fp32_pred = np.argmax(fp32_outputs, axis=1)
    int8_pred = np.argmax(int8_outputs, axis=1)
    agreement = np.mean(fp32_pred == int8_pred)
    fp32_size = os.path.getsize(FP32_MODEL_PATH)
    int8_size = os.path.getsize(INT8_MODEL_PATH)
    print("=== Verification results ===")
    print(f"FP32 model size (bytes): {fp32_size}")
    print(f"INT8 model size (bytes): {int8_size}")
    print(f"Maximum absolute output difference: {max_abs:.6f}")
    print(f"Mean absolute output difference: {mean_abs:.6f}")
    print(f"Predicted class agreement: {agreement:.2%}")

if __name__ == "__main__":
    main()
