from typing import List

import Levenshtein as lvs
import datasets
import nltk
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

ALL_METRICS = dict()


def register(func):
    ALL_METRICS[func.__name__] = func
    return func


@register
def bleu(y: List[str], yhat: List[str], metric=None) -> List[float]:
    result = []
    if metric is None:
        metric = datasets.load_metric('sacrebleu')
    for y_, yhat_ in list(zip(y, yhat)):
        metric.add_batch(predictions=[y_], references=[[yhat_]])
        result.append(metric.compute()['score'])
    return result


@register
def corpus_bleu(y: List[str], yhat: List[str], metric=None) -> float:
    if metric is None:
        metric = datasets.load_metric('sacrebleu')
    metric.add_batch(predictions=yhat, references=[[i] for i in y])
    result = metric.compute()
    return result['score']


@register
def meteor(y: List[str], yhat: List[str]) -> List[float]:
    meteor_scores = [nltk.translate.meteor_score.meteor_score([pred], ref) for (pred, ref) in list(zip(yhat, y))]
    return meteor_scores


@register
def corpus_meteor(y: List[str], yhat: List[str]) -> float:
    meteor_scores = [nltk.translate.meteor_score.meteor_score([pred], ref) for (pred, ref) in list(zip(yhat, y))]
    return np.mean(meteor_scores)


@register
def chrf(y: List[str], yhat: List[str]) -> List[float]:
    chrf_scores = [nltk.translate.chrf_score.sentence_chrf([pred], ref) for (pred, ref) in list(zip(yhat, y))]
    return chrf_scores


@register
def corpus_chrf(y: List[str], yhat: List[str]) -> float:
    chrf_score = nltk.translate.chrf_score.corpus_chrf(y, yhat)
    return chrf_score


@register
def bertscore(y: List[str], yhat: List[str], lang='ru', rescale_with_baseline=True) -> float:
    metric = datasets.load_metric('bertscore')
    metric.add_batch(predictions=yhat, references=[[i] for i in y])
    result = metric.compute(lang=lang, rescale_with_baseline=rescale_with_baseline)
    return result['f1']


def calculate_wer(sequence_a: str, sequence_b: str) -> int:
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))
    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]
    return lvs.distance(''.join(w1), ''.join(w2))


@register
def calculate_wer_corpus(y: List[str], yhat: List[str]) -> List[float]:
    return [calculate_wer(y_, yhat_) for (y_, yhat_) in list(zip(y, yhat))]


def calculate_normalized_wer(sequence_a: str, sequence_b: str) -> float:
    wer = calculate_wer(sequence_a, sequence_b)
    return wer / max(len(sequence_a.split()), len(sequence_b.split()))


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@register
def calculate_paraphrase_similarity(y: List[str], yhat: List[str], device=torch.device('cuda')) -> List[float]:
    result = []
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1').to(device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for (y_, yhat_) in tqdm(list(zip(y, yhat))):
        sentences = [y_, yhat_]
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {i: j.to(device) for i, j in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        output = cos(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
        result.append(output[0].item())
    return result
