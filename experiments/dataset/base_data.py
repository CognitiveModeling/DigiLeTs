from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    @staticmethod
    @abstractmethod
    def default_transform(**kwargs):
        """
        :return: Default transform which should return usable tensors
        """
        return

    @abstractmethod
    def collate_trajectories(self, data):
        """
        Function to collate samples into batches
        :param data:
        :return: Batch tensor
        """
        return



