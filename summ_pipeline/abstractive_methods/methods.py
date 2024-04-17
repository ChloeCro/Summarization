from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap

def bart(text):

    print(text)
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("in bart module: ", summary)
    #formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return summary