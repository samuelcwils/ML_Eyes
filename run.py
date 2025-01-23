from functools import partial
import neptune
import neptune.integrations.optuna as npt_utils
from train import objective
import optuna
from functools import partial

#create a neptune instance for the optuna study
study_run = neptune.init_run(name="optuna-study", project='knightenjoyer15/Project', capture_hardware_metrics=True, api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTE1ZGJhZi1jOTMyLTRiM2QtYTY3MC1jYzJlZTYyYTczODEifQ==")

neptune_callback = npt_utils.NeptuneCallback(study_run)

objective_seeded = partial(objective, seed=5000)
study = optuna.create_study(direction='maximize')
study.optimize(objective_seeded, n_trials=1,callbacks=[neptune_callback], gc_after_trial=True)#gc_after_trial enables garbage collection after each trial

study_run.stop()