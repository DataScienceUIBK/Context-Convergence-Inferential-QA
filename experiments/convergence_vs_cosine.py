import json
import re
import pandas as pd
import tarfile
from statistics import mean
from termcolor import colored
from collections import Counter
from tqdm import tqdm


def compute_em(predicted_answer, ground_truth):
    if predicted_answer == '':
        return 0
    for ga in ground_truth:
        if predicted_answer in ga or ga in predicted_answer:
            return 1
    return 0


def compute_precision(predicted_answer, ground_truths):
    tok = lambda s: re.findall(r"\w+", s.lower())
    max_precision = 0.0

    for gt in ground_truths:
        p = Counter(tok(predicted_answer))
        r = Counter(tok(gt))
        overlap = sum((p & r).values())

        precision = overlap / max(1, sum(p.values()))
        max_precision = max(max_precision, precision)

    return max_precision


def compute_recall(predicted_answer, ground_truths):
    tok = lambda s: re.findall(r"\w+", s.lower())
    max_recall = 0.0

    for gt in ground_truths:
        p = Counter(tok(predicted_answer))
        r = Counter(tok(gt))
        overlap = sum((p & r).values())

        recall = overlap / max(1, sum(r.values()))
        max_recall = max(max_recall, recall)

    return max_recall


def compute_f1(predicted_answer, ground_truths):
    tok = lambda s: re.findall(r"\w+", s.lower())
    max_f1 = 0.0

    for gt in ground_truths:
        p = Counter(tok(predicted_answer))
        r = Counter(tok(gt))
        overlap = sum((p & r).values())

        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / max(1, sum(p.values()))
            recall = overlap / max(1, sum(r.values()))
            f1 = 2 * precision * recall / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1


def analyze(_model, _similarity, _metric):
    metric_map = {'em': compute_em, 'precision': compute_precision, 'recall': compute_recall, 'f1': compute_f1}

    low_score = []
    high_score = []

    with tarfile.open('../dataset/dataset_final.tar.gz', 'r:gz') as tar:
        member = tar.getmember('dataset_final.json')
        f = tar.extractfile(member)
        dataset = json.load(f)

    for category in ['low', 'high']:
        for q in dataset:
            answers = [ans.lower().strip() for ans in q['answers']]
            for order in ['asc_hints', 'desc_hints']:
                for subset in q['subsets'][_similarity][category]:
                    predicted_answer = q['subsets'][_similarity][category][subset][order]['answers'][
                        _model].lower().strip()
                    ground_truth = answers
                    label = metric_map[_metric](predicted_answer, ground_truth)
                    if category == 'low':
                        low_score.append(label)
                    elif category == 'high':
                        high_score.append(label)

    low_score = round(mean(low_score), 4) * 100
    high_score = round(mean(high_score), 4) * 100

    return {f'high_{_metric}': high_score, f'low_{_metric}': low_score}


if __name__ == '__main__':

    for metric in ['em', 'precision', 'recall', 'f1']:
        print(colored(f'{metric}:', color='green', attrs=['bold', 'underline']))
        models = ["llama-32-1b", "gemma-3-1b", "qwen-3-4b", "gemma-3-4b", "qwen-3-8b", "llama-31-8b"]
        for _model in models:
            table_info = dict()
            for _similarity in ['hint_similarity', 'cosine_similarity']:
                em = analyze(_model, _similarity, metric)
                table_info[_similarity] = em
            df = pd.DataFrame.from_dict(table_info, orient='index')
            print(colored(f'{_model}:', color='red', attrs=['bold', 'underline']))
            print(df)

            print()
