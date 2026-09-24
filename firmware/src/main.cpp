#include <Arduino.h>
#include "model_data.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Statically allocated tensor arena (adjust size if needed)
constexpr int kTensorArenaSize = 64 * 1024; // 64KB
static uint8_t tensor_arena[kTensorArenaSize];

void printTensorInfo(const char* name, TfLiteTensor* tensor) {
  Serial.print(name);
  Serial.print(": shape=");
  Serial.print("[");
  for (int i = 0; i < tensor->dims->size; ++i) {
    Serial.print(tensor->dims->data[i]);
    if (i < tensor->dims->size - 1) Serial.print(", ");
  }
  Serial.print("] dtype=");
  Serial.println(tensor->type == kTfLiteInt8 ? "int8" :
               tensor->type == kTfLiteUint8 ? "uint8" :
               tensor->type == kTfLiteFloat32 ? "float32" : "other");
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(1000);
  Serial.println("--- TensorFlow Lite Micro Smoke Test ---");

  static tflite::MicroErrorReporter micro_error_reporter;
  const tflite::Model* model = tflite::GetModel(g_dscnn_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    return;
  }
  Serial.println("MODEL LOADED");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                               kTensorArenaSize, &micro_error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }
  Serial.print("Tensor arena size: ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");

  TfLiteTensor* input = interpreter.input(0);
  printTensorInfo("Input", input);
  // Zero fill input tensor
  if (input->type == kTfLiteInt8 || input->type == kTfLiteUint8) {
    memset(input->data.uint8, 0, input->bytes);
  } else if (input->type == kTfLiteFloat32) {
    memset(input->data.f, 0, input->bytes);
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    Serial.println("INVOKE FAILURE");
    return;
  }
  Serial.println("INVOKE SUCCESS");

  TfLiteTensor* output = interpreter.output(0);
  printTensorInfo("Output", output);
  Serial.print("Output values: ");
  int count = output->bytes / ((output->type == kTfLiteFloat32) ? sizeof(float) : sizeof(int8_t));
  for (int i = 0; i < 3 && i < count; ++i) {
    float val = 0.0f;
    if (output->type == kTfLiteFloat32) {
      val = output->data.f[i];
    } else if (output->type == kTfLiteInt8) {
      val = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    } else if (output->type == kTfLiteUint8) {
      val = (output->data.uint8[i] - output->params.zero_point) * output->params.scale;
    }
    Serial.print(val, 6);
    if (i < 2) Serial.print(", ");
  }
  Serial.println();
}

void loop() {
  delay(1000);
}
