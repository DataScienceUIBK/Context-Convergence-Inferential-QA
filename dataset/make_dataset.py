import json
import random as rnd

rnd.seed(42)

from hinteval import Dataset
from tqdm import tqdm
from itertools import combinations


def convergence(candidates):
    sum_of_ones = 0
    for can in candidates:
        if candidates[can] == 1:
            sum_of_ones += 1
    correct_answer = candidates[list(candidates.keys())[-1]]
    if correct_answer == 1:
        return round(1 - ((sum_of_ones - 1) / len(candidates)), 2)
    else:
        return 0.0


def valid_questions():
    triviahg = Dataset.download_and_load_dataset('triviahg')

    questions = dict()
    all_questions = dict()
    for subset_name in ['training', 'validation', 'test']:
        subset = triviahg[subset_name]
        for q_id in tqdm(subset.get_instance_ids()):
            invalid_question = False
            q = subset.get_instance(q_id)
            q_candidate_answers = q.question.metadata['candidate_answers-llama-3-70b'][:10] + \
                                  [q.question.metadata['candidate_answers-llama-3-70b'][-1]]
            if len(set(q_candidate_answers)) != len(q_candidate_answers):
                continue
            if len(set(q_candidate_answers)) != 11:
                continue

            convs = set()
            for hint in q.hints:
                candidate_answers = hint.metrics['convergence-llm-llama-3-70b'].metadata['scores']
                correct_answer_score = candidate_answers[list(candidate_answers.keys())[-1]]
                if correct_answer_score == 0:
                    invalid_question = True
                convs.add(hint.metrics['convergence-llm-llama-3-70b'].value)
            if not invalid_question:
                questions[q_id] = len(convs)
                all_questions[q_id] = q
    questions = dict(sorted(questions.items(), key=lambda x: x[1], reverse=True))
    return dict([(q_id, all_questions[q_id]) for q_id in list(questions.keys())[:2000]])


def make_dataset(questions):
    final_questions = dict()
    for q_id, q in questions.items():
        q_item = dict()
        q_item['id'] = q_id
        q_item['question'] = q.question.question
        q_item['answers'] = [ans.answer for ans in q.answers]
        q_item['question_type'] = q.question.question_type

        q_candidate_answers = q.question.metadata['candidate_answers-llama-3-70b'][:10] + \
                              [q.question.metadata['candidate_answers-llama-3-70b'][-1]]
        if len(set(q_candidate_answers)) != len(q_candidate_answers):
            continue
        if len(set(q_candidate_answers)) != 11:
            continue

        q_item['candidate_answers'] = q_candidate_answers

        q_hints = []
        for hint in q.hints:
            hint_item = dict()
            hint_item['hint'] = hint.hint
            hint_item['candidate_answers'] = {k: v for k, v in hint.metrics['convergence-llm-llama-3-70b'].metadata[
                'scores'].items() if k in q_candidate_answers}
            hint_item['convergence'] = convergence(hint_item['candidate_answers'])
            q_hints.append(hint_item)
        q_item['hints'] = q_hints
        final_questions[q_id] = q_item

    return final_questions


def generate_orders(hints):
    asc_hints = list(sorted(hints, key=lambda x: x['convergence']))
    desc_hints = list(sorted(hints, key=lambda x: x['convergence'], reverse=True))
    asc_hints = [hint['hint'] for hint in asc_hints]
    desc_hints = [hint['hint'] for hint in desc_hints]
    return {'asc_hints': {'passage': '\n'.join(asc_hints)}, 'desc_hints': {'passage': '\n'.join(desc_hints)}}


def generate_subsets(questions):
    for q_id, q in questions.items():
        hints = q['hints']
        q['subsets'] = dict()
        num_of_hints = len(q['hints'])
        range_list = [str(i) for i in range(num_of_hints)]
        combins = []
        for r in range(3, 6):
            combins.extend(list(combinations(range_list, r)))
        for combin in combins:
            binary = ['0'] * num_of_hints
            combin_hints = []
            for digit in combin:
                binary[int(digit)] = '1'
                combin_hints.append(hints[int(digit)])
            subset_label = ''.join(binary[::-1])
            q['subsets'][subset_label] = generate_orders(combin_hints)

    return questions


def main():
    valid_qs = valid_questions()
    final_questions = make_dataset(valid_qs)
    final_questions_with_subsets = generate_subsets(final_questions)

    random_keys = list(final_questions_with_subsets.keys())
    rnd.shuffle(random_keys)

    final_questions_with_subsets = {k: final_questions_with_subsets[k] for k in random_keys}

    with open('./dataset_base.json', 'w') as f:
        json.dump(final_questions_with_subsets, f, indent=4)


if __name__ == '__main__':
    main()