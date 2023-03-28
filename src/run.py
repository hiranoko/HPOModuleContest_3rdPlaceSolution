import glob
import json
import os
import subprocess
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict

import yaml


def is_json(config_path: Path) -> bool:
    """Evaluates given path is json format or not.

    Args:
        config_path (Path): Path object to be evaluated.

    Returns:
        bool: True is given path is json file.
    """
    return config_path.suffix == ".json"


def load_config(config_path: Path) -> Dict:
    """Loads config file assuming .json, .yaml, or .yml.

    Args:
        config_path (Path): Path object of config file.

    Returns:
        Dict: Dict object of config information.
    """
    with open(config_path, "r") as f:
        if is_json(config_path):
            cfg: Dict = json.load(f)
        else:
            cfg: Dict = yaml.safe_load(f)
    return cfg


def save_config(cfg: Dict, config_path: Path) -> None:
    """Saves config file assuming .json, .yaml, or .yml.

    Args:
        cfg (Dict): Dict to be dumped.
        config_path (Path): Path object of config file.

    Returns:
        None
    """

    with open(config_path, "w") as f:
        if is_json(config_path):
            json.dump(cfg, f)
        else:
            yaml.safe_dump(cfg, f)


def set_configs(test_dir: str, test_function: str, submit_dir: str, search_algorithm: str, trial_number: int):
    # load settings from config file
    target_path = os.path.join(test_dir, test_function)
    try:
        target_cfg: Dict[str, Any] = load_config(Path(f"{target_path}/config.yaml"))
    except FileNotFoundError:
        print("Please create test directory first.")
    assert target_cfg["generic"]["workspace"] == "./work"
    assert target_cfg["generic"]["job_command"] == "python user.py"
    assert target_cfg["resource"]["type"] == "local"
    target_cfg["optimize"]["search_algorithm"] = search_algorithm
    assert trial_number > 0
    target_cfg["optimize"]["trial_number"] = trial_number
    target_cfg["generic"]["params_path"] = os.path.abspath(f"{submit_dir}/model")  # user defined param path

    save_config(target_cfg, Path(f"{target_path}/config.yaml"))


def test_one_function(test_dir: str, test_function: str, submit_dir: str, time_out: int):
    target_path = os.path.join(test_dir, test_function)
    results_path = glob.glob(f"{target_path}/results/*/results.csv")
    print("  {}(previous results: {})".format(test_function, len(results_path)))
    submit_dir = os.path.join(os.path.abspath(submit_dir), "src")
    command = ["bash", "-c", "aiaccel-start --config config.yaml --clean 1>" " stdout.txt 2> stderr.txt"]

    process = subprocess.Popen(command, cwd=os.path.abspath(target_path))
    try:
        process.wait(timeout=time_out)
    except subprocess.TimeoutExpired:
        raise TimeoutError  # タイムアウトエラー

    if process.returncode != 0:
        raise RuntimeError  # 異常終了エラー

    return test_function, len(results_path)


def wrap_test_one_function(params):
    return test_one_function(*params)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--submit-dir", default="./workspace", type=str)
    parser.add_argument("--test-dir", default="./workspace/tests", type=str)
    parser.add_argument("--search-algorithm", default="optimizer.MyOptimizer", type=str)
    parser.add_argument("--trial-number", default=2, type=int)
    parser.add_argument("--time-out", default=360, type=int)
    parser.add_argument("--parallel", default=1, type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print("\nSearch algorithm: {}".format(args.search_algorithm))
    print("Trial number: {}".format(args.trial_number))
    test_functions = os.listdir(args.test_dir)
    print("Number of test functions: {}".format(len(test_functions)))
    for test_function in test_functions:
        print("  {}".format(test_function))
        set_configs(args.test_dir, test_function, args.submit_dir, args.search_algorithm, args.trial_number)

    print("\nStart Execution:")
    start = time.time()
    num_file_exists = {}
    if args.parallel:
        print(" Parallel Processing")
        test_functions = os.listdir(args.test_dir)
        test_functions = [test_func for test_func in test_functions if "_" != test_func[0]]
        params = [(args.test_dir, test_function, args.submit_dir, args.time_out) for test_function in test_functions]
        num_functions = len(test_functions)
        with Pool(num_functions) as p:
            for test_function, num_results in p.imap_unordered(wrap_test_one_function, params):
                num_file_exists[f"{test_function}"] = num_results
    else:
        print(" Simple Processing")
        test_functions = os.listdir(args.test_dir)
        for test_function in test_functions:
            test_function, num_results = test_one_function(test_function, args.submit_dir, args.time_out)
            num_file_exists[f"{test_function}"] = num_results
    with open("./num_file_exists.json", "w", encoding="utf-8") as f:
        json.dump(num_file_exists, f)
    print("\n Done. {} sec".format(time.time() - start))


if __name__ == "__main__":
    main()
