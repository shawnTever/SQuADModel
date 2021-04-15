import torch

import json
from pathlib import Path


# from google.colab import drive
# drive.mount('/content/drive')

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad('dataset/section3.json')
val_contexts, val_questions, val_answers = read_squad('dataset/section3.json')
print()


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters


add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

print()

# from transformers import DistilBertTokenizerFast
#
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#
# train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
# val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

