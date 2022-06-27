__author__ = "Julius Wührer"

import inspect
import os
import time

import torch

from util.logging import Logger
from util.plotting import *
from util.dtw import dtw_distance
from util import extract_vals

from experiments.base import BaseExperiment
from dataset.trajectories_data import TrajectoriesDataset, ToTensor, num_to_char
from models.lstm import LSTM

name = "EfficientLearning"


class EfficientLearning(BaseExperiment):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.setup()

    def run(self):
        """
        Runs the training, or oneshot inference loops saving at specified intervals
        """
        if self.config["load_model"]:
            model_path = self.config["model_path"]

            self.load(model_path, continu=self.config["continue"])

        if self.config["mode"] == "train":
            for epoch in range(self.epoch, self.config["epochs"]):
                self.train(epoch)
                if self.config["test"]:
                    self.test(epoch)
                if epoch % self.config["save_interval"] == 0 or epoch == self.config["epochs"] - 1:
                    self.save(epoch)
        elif self.config["mode"] == "one_shot_inf":
            for epoch in range(self.epoch, self.config["epochs_one_shot_inf"]):
                self.one_shot_inf_mech(epoch)
                if self.config["test"]:
                    self.test(epoch)
                if epoch % self.config["save_interval"] == 0 or epoch == self.config["epochs_one_shot_inf"] - 1:
                    self.save(epoch)
        if self.config["mode"] == "test":
            self.test(0)
        self.end()

    def setup(self):
        """Sets up an experiment.

        Loads the data, initializes a model and optimizer, and sets the
        criterion.

        """
        torch.manual_seed(self.config["seed"])

        self.start_time = time.time()
        self.start_process_time = time.process_time()

        self.device = torch.device(
            "cuda" if self.config["use_cuda"] else "cpu")
        self.one_shot_inf = self.config["mode"] == "one_shot_inf"

        if self.config["load_pretransformed"]:
            transform = ToTensor()
        else:
            transform = TrajectoriesDataset.default_transform()

        self.train_dataset = TrajectoriesDataset(self.config["dataset_path"], input_size=self.config["input_size"],
                                                 participant_size=self.config["participant_size"],
                                                 participant_bounds=self.config["participant_bounds"],
                                                 character_bounds=self.config["character_bounds"],
                                                 instance_bounds=self.config["instance_bounds"],
                                                 include_participant=self.config["include_participant"],
                                                 transform=transform,
                                                 load_pretransformed=self.config["load_pretransformed"])
        self.test_dataset = TrajectoriesDataset(self.config["dataset_path"], input_size=self.config["input_size"],
                                                participant_size=self.config["participant_size"],
                                                participant_bounds=self.config["test_participant_bounds"],
                                                character_bounds=self.config["test_character_bounds"],
                                                instance_bounds=self.config["test_instance_bounds"],
                                                include_participant=self.config["include_participant"],
                                                transform=transform,
                                                load_pretransformed=self.config["load_pretransformed"])
        self.oneshot_dataset = TrajectoriesDataset(self.config["dataset_path"], input_size=self.config["input_size"],
                                                   participant_size=self.config["participant_size"],
                                                   participant_bounds=self.config["oneshot_participant_bounds"],
                                                   character_bounds=self.config["oneshot_character_bounds"],
                                                   instance_bounds=self.config["oneshot_instance_bounds"],
                                                   include_participant=self.config["include_participant"],
                                                   transform=transform,
                                                   load_pretransformed=self.config["load_pretransformed"])
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["batch_size"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.train_dataset.collate_trajectories
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["batch_size"],
            pin_memory=True,
            shuffle=self.config["shuffle_test"],
            collate_fn=self.test_dataset.collate_trajectories
        )

        self.oneshot_loader = torch.utils.data.DataLoader(
            dataset=self.oneshot_dataset,
            batch_size=self.config["oneshot_batch_size"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.oneshot_dataset.collate_trajectories
        )

        # initialize model
        self.model = LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            hidden_bias=self.config["hidden_bias"],
            dropout=self.config["dropout"],
            output_size=self.config["output_size"],
            output_bias=self.config["output_bias"],
            inf=self.one_shot_inf)
        self.model.to(self.device)

        # initialize optimizer
        if self.config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config["lr"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
                betas=(self.config["beta1"], self.config["beta2"]),
                weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.config["lr"], momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"])

        # intialize loss function
        self.criterion = getattr(torch.nn.functional, self.config["criterion"])

        # makes necessary directories
        self.model_directory = self.config["run_directory"] + "/models"
        self.plot_directory = self.config["run_directory"] + "/plots"
        self.log_directory = self.config["run_directory"] + "/logs"
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.plot_directory):
            os.makedirs(self.plot_directory)
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        self.plotter = Plotter(self.config["threshold"], self.config["filetype"])

        # sets up logging
        self.logger = Logger(
            path=self.log_directory, log_interval=self.config["log_interval"], train_len=len(
                self.train_loader.dataset), test_len=len(
                self.test_loader.dataset), one_shot_len=len(self.oneshot_dataset), num_train_batches=len(
                self.train_loader), num_oneshot_batches=len(self.oneshot_loader),
                human_readable=self.config["log_readable"], name=self.config["name"])
        self.logger.log_text("Configuration", str(self.config))
        self.logger.log_text(
            "Dataset", inspect.getsource(
                self.train_dataset.__class__))
        self.logger.log_text("Model", inspect.getsource(self.model.__class__))
        self.logger.log_text("Experiment", inspect.getsource(self.__class__))

        self.epoch = 0
        self.global_step = 0
        self.test_step = 0

    def train(self, epoch):
        """
        One step of the training loop
        :param epoch: Current epoch
        :return:
        """
        self.model.train()
        for batch_idx, (targets, inputs, _, lengths, (label_parts, label_chars, label_insts)) in enumerate(
                self.train_loader):
            tries = 0
            while True:
                try:
                    target_tensors = torch.Tensor(targets)[:, :, :self.config["output_size"]]
                    target_tensors = target_tensors.to(self.device)
                    inputs = inputs.to(self.device)
                    # lengths = lengths.to(self.device) pack_padded_sequence crashes if lengths are on cuda, do not un-
                    # comment until fix is included in Pytorch
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs, None, lengths)

                    loss = self.criterion(outputs, target_tensors)
                    loss.backward()
                    self.optimizer.step()
                    break
                except RuntimeError as e:
                    if ('unspecified launch failure' in str(e) or 'out of memory' in str(e)) and tries < 5:
                        self.logger.log_text("Error", f"Error during training:\n{e}", global_step=self.global_step)
                        print(f"{e}, retrying batch in 5s")
                        time.sleep(5)
                        tries += 1
                        torch.cuda.empty_cache()
                    else:
                        raise e

            batch_size = targets.size(1)

            # plotting every few steps for an overview on
            if self.config["plot_every"] is not None:
                for i in range(self.global_step, self.global_step + batch_size):
                    if i % self.config["plot_every"] == 0:
                        j = i - self.global_step
                        char = num_to_char(label_chars[j])
                        x, y, p, s = extract_vals(outputs, j)
                        tx, ty, tp, ts = extract_vals(targets, j)
                        self.plotter.scatter_multiple(
                            [((x, y, p, s), {"size": 100, "color": plt.get_cmap("viridis"), "marker": "o"}),
                             ((tx, ty, tp, ts),
                              {"size": 100, "color": plt.get_cmap("plasma"), "marker": "D"})],
                            path=self.plot_directory + f"/train_{epoch}_{char}_{i}")

            self.logger.log_train(
                epoch,
                self.global_step,
                batch_idx,
                batch_size,
                loss)

            self.global_step += batch_size

    def test(self, epoch):
        """
        One step of the testing loop (can also be used singularly)
        :param epoch: Current epoch
        :return:
        """
        self.model.eval()
        val_loss = 0
        dtw_average = 0
        dtw_total = np.zeros(self.test_dataset.shape())
        with torch.no_grad():
            for batch_idx, (targets, inputs, inputs_participant, lengths, (label_parts, label_chars, label_insts)) in enumerate(
                    self.test_loader):
                tries = 0
                while True:
                    try:
                        inputs = inputs.to(self.device)
                        target_tensors = targets.to(self.device)[:, :, :self.config["output_size"]]
                        outputs = self.model(inputs, None, lengths)
                        val_loss += self.criterion(outputs, target_tensors)
                        break
                    except RuntimeError as e:
                        if ('unspecified launch failure' in str(e) or 'out of memory' in str(e)) and tries < 5:
                            self.logger.log_text("Error", f"Error during testing:\n{e}", global_step=self.global_step)
                            print(f"{e},  retrying batch in 5s")
                            time.sleep(5)
                            tries += 1
                            torch.cuda.empty_cache()
                        else:
                            raise e

                batch_size = targets.shape[1]

                if self.config["dtw"] or self.config["dtw_total"]:
                    for i in range(batch_size):
                        x, y, p, s = extract_vals(outputs, i)
                        tx, ty, tp, ts = extract_vals(targets, i)
                        stack = np.column_stack((x, y))
                        tstack = np.column_stack((tx, ty))
                        if self.config["dtw_pressure"]:
                            stack = np.column_stack((stack, p))
                            tstack = np.column_stack((tstack, tp))
                        dist = dtw_distance(stack, tstack)
                        if self.config["dtw"]:
                            dtw_average += dist
                        if self.config["dtw_total"]:
                            char = label_chars[i]
                            part = label_parts[i]
                            inst = label_insts[i]
                            dtw_total[part][char][inst] = dist

                if self.config["test_plot_every"] is not None:
                    for i in range(self.test_step, self.test_step + batch_size):
                        if i % self.config["test_plot_every"] == 0:
                            j = i - self.test_step
                            char = num_to_char(label_chars[j])
                            x, y, p, s = extract_vals(outputs, j)
                            tx, ty, tp, ts = extract_vals(targets, j)
                            self.plotter.scatter_multiple(
                                [((x, y, p, s), {"size": 100, "color": plt.get_cmap("viridis"), "marker": "o"}),
                                 ((tx, ty, tp, ts),
                                  {"size": 100, "color": plt.get_cmap("plasma"), "marker": "D"})],
                                path=self.plot_directory + f"/test_{epoch}_{char}_{i}")

                self.test_step += batch_size
        val_loss /= len(self.test_loader)
        dtw_average /= len(self.test_loader)

        if self.config["dtw_total"]:
            dtw_total.dump(os.path.join(self.log_directory, f"{self.config['name']}_dtw_{epoch}.npy"))

        self.logger.log_test(epoch, self.global_step, val_loss, dtw_average)

    def one_shot_inf_mech(self, epoch):
        """
        One step of the one-shot inference loop
        :param epoch: Current epoch
        :return:
        """
        self.model.train()
        for batch_idx, (targets, inputs, _, lengths, (label_parts, label_chars, label_insts)) in enumerate(
                self.oneshot_loader):
            tries = 0
            while True:
                try:
                    target_tensors = torch.Tensor(targets)[:, :, :self.config["output_size"]]
                    target_tensors = target_tensors.to(self.device)

                    inputs = inputs.to(self.device)
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs, None, lengths)
                    loss = self.criterion(outputs, target_tensors)
                    loss.backward()
                    self.optimizer.step()
                    break
                except RuntimeError as e:
                    if ('unspecified launch failure' in str(e) or 'out of memory' in str(e)) and tries < 5:
                        self.logger.log_text("Error", f"Error during oneshot:\n{e}", global_step=self.global_step)
                        print(f"{e},  retrying batch in 5s")
                        time.sleep(5)
                        tries += 1
                        torch.cuda.empty_cache()
                    else:
                        raise e

            batch_size = targets.size(1)

            self.logger.log_one_shot(
                epoch,
                self.global_step,
                batch_idx,
                batch_size,
                loss)

            if self.config["plot_every"] is not None:
                for i in range(self.global_step, self.global_step + batch_size):
                    if i % self.config["plot_every"] == 0:
                        j = i - self.global_step
                        char = num_to_char(label_chars[j])
                        x, y, p, s = extract_vals(outputs, j)
                        tx, ty, tp, ts = extract_vals(targets, j)
                        self.plotter.scatter_multiple(
                            [((x, y, p, s), {"size": 100, "color": plt.get_cmap("viridis"), "marker": "o"}),
                             ((tx, ty, tp, ts),
                              {"size": 100, "color": plt.get_cmap("plasma"), "marker": "D"})],
                            path=self.plot_directory + f"/train_{epoch}_{char}_{i}")

            self.global_step += batch_size


    def load(self, model_path, continu=False):
        """
        Loads a model from a save file. Only loads the optimizer if we want to continue training with the same
        parameters (e.g. after a crash).
        :param model_path:
        :param continu:
        :return:
        """
        dic = torch.load(model_path)
        if continu:
            self.epoch = dic['epoch']
            self.global_step = dic['global_step']
            self.test_step = dic['test_step']
            self.optimizer.load_state_dict(dic['optimizer_state_dict'])
        self.model.load_state_dict(dic['model_state_dict'])

    def save(self, epoch):
        """
        Saves current epoch, gloabl step, model and optimizer to file
        :param epoch:
        :return:
        """
        # save epoch, global_step, model, and optimizer to file
        path = os.path.join(
            self.model_directory,
            self.config["name"] +
            "_epoch" +
            str(epoch) +
            ".pth")
        torch.save({"epoch": epoch + 1,
                    "global_step": self.global_step + 1,
                    "test_step": self.test_step + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   path)

    def end(self):
        """
        Calculates total time and appends it to the log
        :return:
        """
        self.end_time = time.time()
        self.end_process_time = time.process_time()
        self.time = float(self.end_time - self.start_time)
        self.process_time = float(
            self.end_process_time -
            self.start_process_time)
        self.logger.log_text("Time", str(self.time))
        self.logger.log_text("Process time", str(self.process_time))
        self.logger.close()
