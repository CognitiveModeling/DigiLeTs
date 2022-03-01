from util import openslice, multislice

config = {
    "name": "Visualize",
    "experiment_name": "EfficientLearning",
    "dataset_path": "./data/transformed/cogmod.pickle",
    "load_pretransformed": True,
    "run_directory": "./runs/efficient/visualize",
    "plot_directory": "images",
    "load_model": True,
    "model_path": "./runs/efficient/models/Oneshot_epoch999.pth",
    "continue": False,
    "mode": "test",
    "infer_target": "both",
    "test": True,
    "dtw": False,
    "dtw_total": False,
    "epochs": 1,
    "epochs_one_shot_inf": 1000,
    "batch_size": 1,
    "oneshot_batch_size": 4,
    "shuffle_test": False,
    "log_interval": 1,
    "log_readable": False,
    "save_interval": 5,
    "plot_every": 10100,
    "test_plot_every": 1,
    "threshold": 0.90,
    "filetype": "png",
    "input_size": 62,
    "participant_size": 77,
    "include_participant": True,
	"average_participant": False,
	"cycle_participant": None,
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
    "participant_bounds": multislice([(0, 75, 1)]),
    "character_bounds": multislice([(10, 23, 1)]),
    "instance_bounds": openslice(),
    "test_participant_bounds": openslice(),
    "test_character_bounds": multislice([(10, 36, 1)]),
    "test_instance_bounds": multislice([(2, 3, 1)]),
    "oneshot_participant_bounds": multislice([(0, 1, 1)]),
    "oneshot_character_bounds": multislice([(22, 36, 1)]),
    "oneshot_instance_bounds": multislice([(0, 1, 1)]),
    "instances": None
}
