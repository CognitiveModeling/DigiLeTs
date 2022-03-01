import os
import json

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """
    Logging utility class, logs both to console as well as a tensorboard SummaryWriter
    """

    def __init__(self, path, log_interval, train_len,
                 test_len, one_shot_len, num_train_batches, num_oneshot_batches, num_test_batches=1, human_readable=False, name='human-readable'):
        self.log_interval = log_interval
        self.train_len = train_len
        self.test_len = test_len
        self.one_shot_len = one_shot_len
        self.num_train_batches = num_train_batches
        self.num_oneshot_batches = num_oneshot_batches
        self.num_test_batches = num_test_batches
        self.path = path
        self.writer = SummaryWriter(log_dir=path)
        self.human_readable = human_readable
        self.name = name
        self.results = {
            'Loss/Train': [],
            'DTW/Train': [],
            'Loss/Test': [],
            'DTW/Test': [],
            'Text': []
        }

    def log_text(self, name, text, global_step=None):
        self.results['Text'].append((name, text, global_step))
        self.writer.add_text(name, text, global_step=global_step)

    def log_train(self, epoch, global_step, batch_idx, batch_size, loss, dtw=0):
        if batch_idx % self.log_interval == 0:
            self.results['Loss/Train'].append((loss.item(), global_step))
            self.results['DTW/Train'].append((dtw, global_step))
            self.writer.add_scalar(
                'Loss/Train', loss.item(),
                global_step=global_step)
            print(
                'Train Epoch: {} [{:5d}/{:5d} samples ({:2.0f}%)]\t Batch Loss: {:.6f}\t Batch DTW: {:.6f}'. format(
                    epoch,
                    batch_idx *
                    batch_size,
                    self.train_len,
                    100. *
                    batch_idx /
                    self.num_train_batches,
                    loss.item(),
                    dtw))

    def log_test(self, epoch, global_step, val_loss, dtw=0):
        self.results['Loss/Test'].append((val_loss.item(), global_step))
        self.results['DTW/Test'].append((dtw, global_step))
        self.writer.add_scalar(
            'Loss/Test',
            val_loss.item(),
            global_step=global_step)
        self.writer.add_scalar(
            'DTW/Test',
            dtw,
            global_step=global_step
        )
        print(
            'Test  Epoch: {} [{:5d} samples]\t\t\t Loss: {:.6f} \t\t DTW: {:.6f}'.format(
                epoch,
                self.test_len,
                val_loss.item(),
                dtw))

    def log_one_shot(self, epoch, global_step, batch_idx, batch_size, loss):
        if batch_idx % self.log_interval == 0:
            self.results['Loss/Train'].append((loss.item(), global_step))
            self.writer.add_scalar(
                'Loss/Train', loss.item(),
                global_step=global_step)
            print(
                'Oneshot Epoch: {} [{:5d}/{:5d} samples ({:2.0f}%)]\t Batch Loss: {:.6f}'.format(
                    epoch,
                    batch_idx *
                    batch_size,
                    self.one_shot_len,
                    100. *
                    batch_idx /
                    self.num_oneshot_batches,
                    loss.item()))

    def close(self):
        # write machine readable summary as well
        if self.human_readable:
            with open(os.path.join(self.path, f"{self.name}.json"), 'w') as f:
                json.dump(self.results, f, indent=4)

        self.writer.close()
