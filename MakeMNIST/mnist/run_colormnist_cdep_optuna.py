from run_mnist_method_optuna_core import main_fixed


if __name__ == "__main__":
    main_fixed(dataset_name="color", method_name="cdep", default_study_name="colormnist_cdep_optuna")
