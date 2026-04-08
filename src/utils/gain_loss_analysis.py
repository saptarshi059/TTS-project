from argparse import ArgumentParser
from evaluate import load
from tqdm import tqdm
import pandas as pd


def main(dataset:str):
    def check_for_null(ans_str):
        return "" if ans_str is None else ans_str

    def check_for_correctness(pred, gold):
        pred_list = [{'prediction_text': pred, 'id': str(0), 'no_answer_probability': 0.}]
        ref_list = [{'answers': {'answer_start': [0], 'text': [gold]}, 'id': str(0)}]

        res = squad_metric.compute(predictions=pred_list, references=ref_list)
        return True if res['exact'] == 100.0 else False

    squad_metric = load("squad_v2")
    results_df = pd.read_json(
        f"../../experiment_runs/main_framework_run/{dataset}/system2/final_response/final_responses.jsonl",
        lines=True)

    stats = {}
    for row in tqdm(results_df.itertuples()):
        sys_one_ans = check_for_null(row.system_1_guess)
        sys_two_ans = check_for_null(row.final_ans)
        gold_ans = row.answer

        sys_one_ans_correct = check_for_correctness(sys_one_ans, gold_ans)
        sys_two_ans_correct = check_for_correctness(sys_two_ans, gold_ans)

        # Both Correct
        if sys_one_ans_correct and sys_two_ans_correct:
            stats['both'] = 1 + stats.get('both', 0)

        # Gain (Incorrect -> Correct)
        elif sys_one_ans_correct is False and sys_two_ans_correct is True:
            stats['gain'] = 1 + stats.get('gain', 0)

        # Loss (Correct -> Incorrect)
        elif sys_one_ans_correct is True and sys_two_ans_correct is False:
            stats['loss'] = 1 + stats.get('loss', 0)

    stats['net_gain'] = stats['gain'] - stats['loss']

    print(f"Saving stats for {dataset}...")
    pd.DataFrame([stats]).to_csv(f"{dataset}_stats.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(args.dataset)
