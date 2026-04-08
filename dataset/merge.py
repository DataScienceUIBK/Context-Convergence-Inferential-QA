import json
from termcolor import colored
import os

def update():
    print(colored(f'Merging the generated answers from the models to a file...', 'yellow'))
    print()
    with open(f'./dataset_similarities.json', mode='r') as f:
        dataset = json.load(f)
    dataset_dict = dict()
    for q in dataset:
        dataset_dict[q['id']] = q
    models_name = [file_name.split('_')[1][:-5] for file_name in os.listdir(f'./qa_results')]
    for model in models_name:
        with open(f'./qa_results/answers_{model}.json') as f:
            answers = json.load(f)
        for _answer in answers:
            q_id, sim_method, category, subset, order, answer = _answer
            if 'answers' not in dataset_dict[q_id]['subsets'][sim_method][category][subset][order]:
                dataset_dict[q_id]['subsets'][sim_method][category][subset][order]['answers'] = dict()
            dataset_dict[q_id]['subsets'][sim_method][category][subset][order]['answers'][model.lower()] = answer
    dataset_new = []
    for item in dataset_dict.values():
        dataset_new.append(item)
    with open(f'./dataset_final.json', mode='w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    update()