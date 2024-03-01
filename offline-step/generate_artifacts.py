from onnxruntime.training import artifacts
import torch
import onnx
import transformers


transformers_model = transformers.LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ignore_mismatched_sizes=True)
tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")


class FlatModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *local_inputs):
        return self.model(inputs.input_ids, inputs.attention_mask)

# model = FlatModel()
model = transformers_model
input_names = ["input_ids", "attention_mask", "position_ids"]
output_names = ["loss", "logits"]

torch.onnx.export(model,
                  (inputs["input_ids"], inputs["attention_mask"]),
                  "tinyllama_full_layer2.onnx",
                  input_names = input_names, 
                  output_names = output_names,
                  export_params=True,
                  opset_version=14,
                  training=torch.onnx.TrainingMode.TRAINING,
                  do_constant_folding=False,
                  dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                  }
                  )

onnx_model_path = "tinyllama_full_layer2.onnx"
# onnx_model_path = "mnist.onnx"
onnx_model = onnx.load(onnx_model_path)
requires_grad = [param.name for param in onnx_model.graph.initializer] # if param.name not in requires_grad]
frozen_params = []
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    loss=artifacts.LossType.MSELoss,
    artifact_directory="artifacts_generated_full",
    optimizer=artifacts.OptimType.AdamW,
    ort_format=False,
    loss_input_names=["loss"]
)