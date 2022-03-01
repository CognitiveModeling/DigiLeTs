from util import openslice, multislice

config = {
    "name": "Oneshot",
    "experiment_name": "ExtendingOmniglot",
    "dataset_path": "./data/transformed/cogmod.pickle",
    "load_pretransformed": True,
    "run_directory": "./runs/extomni/oneshot",
    "plot_directory": "images",
    "load_model": True,
    "model_path": "./runs/extomni/base/models/Base_epoch199.pth",
    "continue": False,
    "mode": "one_shot_inf",
    "infer_target": "both",
    "test": False,
    "dtw": False,
    "dtw_total": False,
    "epochs": 200,
    "epochs_one_shot_inf": 1000,
    "batch_size": 1,
    "oneshot_batch_size": 1,
    "shuffle_test": True,
    "log_interval": 128,
    "log_readable": False,
    "save_interval": 4,
    "plot_every": None,
    "test_plot_every": None,
    "threshold": 0.90,
    "filetype": "png",
    "input_size": 62,
    "participant_size": 77,
    "include_participant": True,
    "hidden_size": 100,
    "embedding_size": 50,
    "num_layers": 1,
    "hidden_bias": False,
    "dropout": 0,
    "output_size": 4,
    "output_bias": False,
    "criterion": "mse_loss",
    "optimizer": "adam",
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0,
    "seed": 12345678,
    "use_cuda": True,
    "participant_bounds": multislice([(0, 1, 1)]),
    "character_bounds": multislice([(10, 36, 1)]),
    "instance_bounds": openslice(),
    "test_participant_bounds": multislice([(1, 2, 1)]),
    "test_character_bounds": multislice([(23, 36, 1)]),
    "test_instance_bounds": multislice([(1, None, 1)]),
    "oneshot_participant_bounds": multislice([(66, 77, 1)]),
    "oneshot_character_bounds": multislice([(23, 36, 1)]),
    "oneshot_instance_bounds": multislice([(2, 3, 1)]),
    "instances": None
}