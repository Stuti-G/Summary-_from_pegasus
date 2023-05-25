from datasets import load_dataset
from transformers import pipeline, set_seed

summaries = {}

dataset = load_dataset("cnn_dailymail", version="3.0.0")

sample_text = dataset["train"][1]["article"][:1000]

pipe = pipeline('summarization', model="google/pegasus-cnn_dailymail")

pipe_out = pipe(sample_text)

summaries["pegasus"] = pipe_out[0]["summary_text"].replace(" .<n>", ".\n")

print("GROUND TRUTH")

print(dataset['train'][1]['highlights'])


for model_name in summaries:
    print(model_name.upper())
    print(summaries[model_name])
