import optuna

storage_name = "sqlite:///{}.db".format("mystudy")
study = optuna.create_study(study_name="mystudy", direction='minimize', storage=storage_name, load_if_exists=True)