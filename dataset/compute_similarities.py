import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from statistics import mean
from copy import deepcopy


def compute_hint_similarity(hints):
    similarities = []
    for hint in hints:
        similarities.append(hint['convergence'])
    return similarities


def compute_cosine_similarity(question, hints):
    similarities = []
    hints_list = [hint['hint'] for hint in hints]
    questions_list = [question]

    q_embeddings = model.encode(questions_list)
    h_embeddings = model.encode(hints_list)

    scores = model.similarity(q_embeddings, h_embeddings)
    for score in scores.squeeze():
        similarities.append(round(score.item(), 4))
    return similarities


def compute_final_similarities(similarities, subset):
    one_idxs = ''
    for idx, c in enumerate(subset[::-1]):
        if c == '1':
            one_idxs += str(idx)
    sims = [similarities[int(idx)] for idx in one_idxs]
    return round(mean(sims), 4)


def main():
    with open('./dataset_base.json', 'r') as f:
        dataset = json.load(f)

    final_dataset = []
    for q_id, q in tqdm(dataset.items()):
        question = q['question']
        hints = q['hints']
        hint_similarity = compute_hint_similarity(hints)
        cosine_similarity = compute_cosine_similarity(question, hints)

        subsets = dict()
        for _subset in q['subsets']:
            q['subsets'][_subset]['similarities'] = {'hint_similarity': 0, 'cosine_similarity': 0}
            q['subsets'][_subset]['similarities']['hint_similarity'] = compute_final_similarities(hint_similarity,
                                                                                                  _subset)
            q['subsets'][_subset]['similarities']['cosine_similarity'] = compute_final_similarities(cosine_similarity,
                                                                                                    _subset)
            subsets[_subset] = q['subsets'][_subset]['similarities']

        new_subsets = {'hint_similarity': {}, 'cosine_similarity': {}}
        for sims in ['hint_similarity', 'cosine_similarity']:
            subsets_based_on_similarity = sorted(subsets.items(), key=lambda x: x[1][sims])
            ## LOW
            low_group = list(dict(subsets_based_on_similarity[:10]).keys())
            new_subsets[sims]['low'] = {}
            for _subset in low_group:
                new_subsets[sims]['low'][_subset] = deepcopy(q['subsets'][_subset])
                new_subsets[sims]['low'][_subset]['similarity'] = q['subsets'][_subset]['similarities'][sims]
                del new_subsets[sims]['low'][_subset]['similarities']
            ## HIGH
            high_group = list(dict(subsets_based_on_similarity[-10:]).keys())
            new_subsets[sims]['high'] = {}
            for _subset in high_group:
                new_subsets[sims]['high'][_subset] = deepcopy(q['subsets'][_subset])
                new_subsets[sims]['high'][_subset]['similarity'] = q['subsets'][_subset]['similarities'][sims]
                del new_subsets[sims]['high'][_subset]['similarities']

        q['subsets'] = new_subsets
        final_dataset.append(q)

    with open('./dataset_similarities.json', 'w') as f:
        json.dump(final_dataset, f, indent=4)


if __name__ == '__main__':
    model = SentenceTransformer("all-mpnet-base-v2", model_kwargs={"torch_dtype": "bfloat16"})
    main()
