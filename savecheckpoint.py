import optuna
import joblib

#callback for optuna
class SaveCheckpoint:
    def __init__(self, name):
        self.name = name

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        joblib.dump(study, self.name + "_checkpoint.pkl")