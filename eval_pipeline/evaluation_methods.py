import evaluate
from evaluate import load

def rouge_method(references, predictions):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=[predictions], references=[references])
    return results

def bertscore(references, predictions):
    bertscore = load('bertscore')
    results = bertscore.compute(predictions=[predictions], references=[references], lang="nl")
    results.pop('hashcode', None)
    return results

def llm_method(references, predictions):
    pass