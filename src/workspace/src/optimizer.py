import json
import os
from typing import Optional

import optuna
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer.tpe_optimizer import create_distributions
from botorch.settings import suppress_botorch_warnings, validate_input_scaling
from candidate import turbo_candidates_func as custom_candidates_func
from sampler import BoTorchSampler

suppress_botorch_warnings(True)
validate_input_scaling(False)


class MyOptimizer(AbstractOptimizer):
    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.parameter_pool = {}
        self.parameter_list = []
        self.study = None
        self.distributions = None
        self.trial_pool = {}

        # custom
        params_path = self.config.config.get("generic", "params_path")
        with open(os.path.join(params_path, "test.json")) as f:
            self.CFG = json.load(f)
        self.study_name = self.CFG["study_name"]

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes."""
        super().pre_process()
        self.parameter_list = self.params.get_parameter_list()
        self.create_study()
        self.distributions = create_distributions(self.params)

    def post_process(self) -> None:
        """Post-procedure after executed processes."""
        self.check_result()
        super().post_process()

    def check_result(self) -> None:
        """Check the result files and add it to sampler object."""
        del_keys = []
        for trial_id, _ in self.parameter_pool.items():
            objective = self.storage.result.get_any_trial_objective(trial_id)
            if objective is not None:
                trial = self.trial_pool[trial_id]
                self.study.tell(trial, objective)
                del_keys.append(trial_id)

        for key in del_keys:
            self.parameter_pool.pop(key)
            self.logger.info(f"trial_id {key} is deleted from parameter_pool")

        self.logger.debug(f"current pool {[k for k, v in self.parameter_pool.items()]}")

    def is_startup_trials(self) -> bool:
        return self.num_of_generated_parameter < self.study.sampler._n_startup_trials

    @staticmethod
    def print_parameters(new_params: dict, message: str):
        """Print DataFrame for debugging parameters"""
        import pandas as pd

        columns, values = [], []
        for param in new_params:
            columns.append(param["parameter_name"])
            values.append(param["value"])
        print(message, pd.DataFrame(data=[values], columns=columns), sep="\n")

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.
        """
        self.check_result()
        self.logger.debug(f"number: {number}, pool: {len(self.parameter_pool)} losses")

        if (not self.is_startup_trials()) and (len(self.parameter_pool) >= 1):
            return None

        if len(self.parameter_pool) >= self.config.num_node.get():
            return None

        new_params = []
        trial = self.study.ask(self.distributions)

        for param in self.params.get_parameter_list():
            new_param = {"parameter_name": param.name, "type": param.type, "value": trial.params[param.name]}
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f"newly added name: {trial_id} to parameter_pool")

        self.print_parameters(new_params, "Generated:")  # debug
        return new_params

    def generate_initial_parameter(self):

        enqueue_trial = {}
        for hp in self.params.hps.values():
            if hp.initial is not None:
                enqueue_trial[hp.name] = hp.initial

        # all hp.initial is None
        if len(enqueue_trial) == 0:
            return self.generate_parameter()

        self.study.enqueue_trial(enqueue_trial)
        trial = self.study.ask(self.distributions)

        new_params = []
        for name, value in trial.params.items():
            new_param = {"parameter_name": name, "type": self.params.hps[name].type, "value": value}
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f"newly added name: {trial_id} to parameter_pool")

        self.print_parameters(new_params, "Initial:")  # debug
        return new_params

    def create_study(self) -> None:
        """Create the optuna.study object and store it."""
        if self.study is None:
            custom_sampler = BoTorchSampler(
                n_startup_trials=self.CFG["n_startup_trials"],
                seed=self.CFG["seed"],
                candidates_func=custom_candidates_func,
                acqf=self.CFG["acqf"],
            )
            self.study = optuna.create_study(
                sampler=custom_sampler,
                study_name=self.study_name,
                direction=self.config.goal.get().lower(),
            )
