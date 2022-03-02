__author__ = "Julius Wührer"

import argparse
import pkgutil
import importlib.util
import sys

import torch

import experiments
from util.config_iterator import ConfigIterator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Begin an experiment with the Styling Trajectories architecture, "
                                     "use without arguments to start interactive prompt.")
    parser.add_argument('config_path')
    parser.add_argument('--overwrite_config', '-o', nargs="*", default=[])
    args = parser.parse_args()

    print("Loading config")
    spec = importlib.util.spec_from_file_location("config", args.config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = module.config

    if not isinstance(configs, ConfigIterator):
        print(f"Overwriting config with specified values\n{args.overwrite_config}")
        for pair in args.overwrite_config:
            key, value = pair.split('=')
            # try converting to int
            try:
                value = int(value)
            except ValueError:
                # try converting to bool
                if value == "True":
                    value = True
                elif value == "False":
                    value = False

            configs[key] = value
        configs = [configs]

    print("Loading experiments")
    experiment_modules = [importlib.import_module(modname) for importer, modname, ispkg in
                       pkgutil.iter_modules(experiments.__path__, prefix="experiments.") if not ispkg]
    experiment_modules = [module for module in experiment_modules if "name" in dir(module)]

    for config in configs:
        print(f"Instantiating experiment {config['name']}")
        experiment_module = None
        for module in experiment_modules:
            if config["experiment_name"] == module.name:
                experiment_module = module
        experiment_class = getattr(experiment_module, experiment_module.name)
        experiment_instance = experiment_class(config)

        print("Running experiment")
        experiment_instance.run()
        print("Finished experiment")
        print("Freeing up resources")
        del experiment_instance
        torch.cuda.empty_cache()
    sys.exit(0)
