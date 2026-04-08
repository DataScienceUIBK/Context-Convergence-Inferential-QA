import json
import tarfile
import pandas as pd
from statistics import mean
from termcolor import colored

def analyze(_model, _similarity):
    low_em = {'asc_hints': [], 'desc_hints': []}
    high_em = {'asc_hints': [], 'desc_hints': []}

    with tarfile.open('../dataset/dataset_final.tar.gz', 'r:gz') as tar:
        member = tar.getmember('dataset_final.json')
        f = tar.extractfile(member)
        dataset = json.load(f)

    for category in ['low', 'high']:
        for q in dataset:
            answers = [ans.lower().strip() for ans in q['answers']]
            for order in ['asc_hints', 'desc_hints']:
                for subset in q['subsets'][_similarity][category]:
                    label = int(q['subsets'][_similarity][category][subset][order]['answers'][
                                    _model].lower().strip() in answers)
                    if category == 'low':
                        low_em[order].append(label)
                    elif category == 'high':
                        high_em[order].append(label)

    for order in ['asc_hints', 'desc_hints']:
        low_em[order] = round(mean(low_em[order]),4)*100
        high_em[order] = round(mean(high_em[order]),4)*100

    return {'low_em': low_em, 'high_em': high_em}


if __name__ == '__main__':
    models = ["gemma-3-1b", "llama-32-1b", "qwen-3-4b", "gemma-3-4b", "qwen-3-8b", "llama-31-8b"]
    for _model in models:
        table_info = dict()
        for _similarity in ['hint_similarity']:
            em = analyze(_model, _similarity)
            table_info[_similarity] = em
        df = pd.concat({k: pd.DataFrame(v).T for k, v in table_info.items()}, axis=1)
        print(colored(f'{_model}:', color='red', attrs=['bold', 'underline']))
        print(df)

        print()
