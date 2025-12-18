from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from IPython.display import display
import json
# torch._inductor.config.compile_threads = 1
# torch._inductor.config.triton.cudagraphs = False
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.reset()
# torch._dynamo.disable()

# Check for available devices
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU")
model_name = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

if __name__=="__main__":

  model = model.to(device)

  # --- Count total and trainable parameters ---
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  dtype_size = {
      torch.float32: 4,
      torch.float16: 2,
      torch.bfloat16: 2,
      torch.int8: 1
  }
  param_size_bytes = sum(
      p.numel() * dtype_size.get(p.dtype, 4)
      for p in model.parameters()
  )
  size_in_mb = param_size_bytes / (1024 ** 2)
  size_in_gb = size_in_mb / 1024

  print(f"Model: {model_name}")
  print(f"Total parameters: {total_params:,}")
  print(f"Trainable parameters: {trainable_params:,}")
  print(f"Approx. size: {size_in_mb:.2f} MB ({size_in_gb:.2f} GB)")

  p = next(model.parameters())
  p.dtype

  with open('/gdrive/MyDrive/AIDER_VQA_results_GPT4.json', 'r') as file:
      data = json.load(file)

  # Questions to ask the model for each image
  questions = [
      "Classify the incident out of possible classes (collapsed_building, fire, flood, normal, traffic_incident)",
      "Damage level? (none / low / medium / high)",
      "Identify damaged objects?",
      "Human presence? (none / visible / uncertain)",
      "Give a small 50 words analysis of scene?"
  ]

  answers_obj = {
      "incident_class": "",
      "damage_level": "",
      "damaged_objects": [],
      "human_presence": "",
      "scene_analysis": ""
    }

  with open('/gdrive/MyDrive/AIDER_VQA_results_moondream-1B.json', 'r') as file:
    results = json.load(file)
    print(len(results))

  image_files = list(data.keys())
  for idx, imagepath in enumerate(image_files[len(results):]):
    print(f"processing file {idx}")
    answers_obj = {
      "incident_class": "",
      "damage_level": "",
      "damaged_objects": [],
      "human_presence": "",
      "scene_analysis": ""
    }
    image = Image.open(f"/gdrive/MyDrive/{imagepath}")
    answers_obj["incident_class"] = model.query(image, questions[0])["answer"]
    answers_obj["damage_level"] = model.query(image, questions[1])["answer"]
    answers_obj["human_presence"] = model.query(image, questions[3])["answer"]
    answers_obj["scene_analysis"] = model.query(image, questions[4])["answer"]
    detections = []
    for obj in data[imagepath]['damaged_objects']:
      objects = model.detect(image, f"damaged {obj}")["objects"]
      if len(objects) > 0:
        detections.append(1)
      else:
        detections.append(0)
    answers_obj["damaged_objects"] = detections
    results.append(answers_obj)

    with open('DisasterVQA_results_moondream-2B.json', 'w') as file:
      json.dump(results, file, indent=4)

