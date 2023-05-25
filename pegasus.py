from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline


def pegasus_test(example_text):
    model_name = "google/pegasus-xsum"
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

    tokens = pegasus_tokenizer(
        example_text, truncation=True, padding="longest", return_tensors="pt")

    encoded_summary = pegasus_model.generate(**tokens)

    decoded_summary = pegasus_tokenizer.decode(
        encoded_summary[0], skip_special_tokens=True)

    # print(decoded_summary)

    summarizer = pipeline("summarization", model=model_name,
                          tokenizer=pegasus_tokenizer, framework="pt")

    summary = summarizer(example_text, min_length=30)

    summary_final = summary[0]["summary_text"]

    return summary_final, example_text, len(example_text), len(summary_final)
