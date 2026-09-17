from datasets import Dataset

data = [
    {"text": "Explain knowledge distillation."},
    {"text": "What is a transformer?"}
]

ds = Dataset.from_list(data)
ds.save_to_disk("./dummy_dataset")