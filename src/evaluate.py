import glob
import json
import os
from argparse import ArgumentParser

import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test-dir", default="./workspace/tests", type=str)
    parser.add_argument("--trial-number", default=2, type=int)
    parser.add_argument("--run-result-path", default="./num_file_exists.json", type=str)
    args = parser.parse_args()

    return args


def main():
    print("\nEvaluation:")
    args = parse_args()
    test_functions = os.listdir(args.test_dir)
    test_functions = [test_func for test_func in test_functions if "_" != test_func[0]]
    with open(args.run_result_path) as f:
        num_file_exists = json.load(f)
    for test_function in test_functions:
        print("\n {}".format(test_function))
        results_path = glob.glob(f"{args.test_dir}/{test_function}/results/*/results.csv")

        if len(results_path) == num_file_exists[test_function]:
            print("   No update")
        else:
            print("   Update {} -> {}".format(num_file_exists[test_function], len(results_path)))

        # データの読み込み
        for result_path in results_path:
            results = np.loadtxt(result_path, skiprows=1, delimiter=",")

            # 最適値の追跡
            current_best = np.inf
            best = []
            for xi in results:
                current_best = min(xi[-1], current_best)
                best.append(current_best)
            best = np.array(best)

            # スコアの計算
            score = best[int(args.trial_number / 2) :].sum()

            # 結果の表示
            print("   result: {}".format(result_path))
            print(f"    best objective: {best[-1]}")
            print(f"    score:          {score}")


if __name__ == "__main__":
    main()
