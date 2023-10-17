import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

import utils

DATASET_FOLDER = "./graph_data"


class AirQualityClassification(InMemoryDataset):
    """
    Our graph for the air quality data (classification task)
    """

    def __init__(
        self,
        seed,
        train_ratio=0.3,
        val_ratio=0.3,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        data, self.slices = torch.load(self.processed_paths[0])
        self.data = Data.from_dict(data) if isinstance(data, dict) else data
        self.mask_tr, self.mask_va, self.mask_te = self.splits(data, seed, train_ratio, val_ratio)

    @property
    def processed_dir(self):
        return os.path.join(DATASET_FOLDER, self.__class__.__name__)

    @property
    def processed_file_names(self):
        return "data_processed.pt"

    def splits(self, data, seed, train_ratio, val_ratio):
        paths = [
            os.path.join(
                self.processed_dir,
                f"mask_seed{seed}_train{str(train_ratio).replace('.', 'p')}_val{str(val_ratio).replace('.', 'p')}_{name}.npy",
            )
            for name in ["tr", "va", "te"]
        ]
        if any(not os.path.exists(path) for path in paths):
            for path, mask in zip(paths, utils.get_masks(data.y, seed, train_ratio, val_ratio)):
                np.save(path, mask)

        return [np.load(path) for path in paths]

    def process(self):
        """
        Called if any of self.processed_file_names don't exist in folder self.processed_dir
        """

        # load data
        inputs = utils.open_pm25()
        inputs = inputs.reshape(-1, inputs.shape[-1])  # flatten in space, keep time (i.e. last dimension)
        outputs = utils.open_land_cover().flatten()  # the land cover classes

        # don't use pixels in graph that don't have inputs or where the output is nan (i.e. over sea)
        mask = ~np.all(np.isnan(inputs), axis=1) * ~np.isnan(outputs)
        outputs = outputs[mask]
        inputs = inputs[mask]

        # normalize inputs
        ip_min, ip_max = np.nanmin(inputs), np.nanmax(inputs)
        inputs = (inputs - ip_min) / (ip_max - ip_min)

        # the 'pos' attribute is used by pre_transform=T.KNNGraph(k=10, force_undirected=True) to create undirected graph
        data_list = [
            Data(
                pos=torch.tensor(inputs, dtype=torch.float),
                x=torch.tensor(inputs, dtype=torch.float),
                y=torch.tensor(
                    outputs - 1, dtype=torch.long
                ),  # subtract 1 to make classes start at 0, type long because it's a class label
            )
        ]

        # from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html#creating-in-memory-datasets
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # saving a huge Python list is slow so it gets collated into one huge Data object via torch_geometric.data.InMemoryDataset.collate()
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def read(self):
        data = torch.load(os.path.join(self.path, "data.pt"))
        return data


class AirQualityRegression(InMemoryDataset):
    """
    Our graph for the air quality data (regression task)
    """

    def __init__(
        self,
        seed,
        train_ratio=0.3,
        val_ratio=0.3,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        data, self.slices = torch.load(self.processed_paths[0])
        self.data = Data.from_dict(data) if isinstance(data, dict) else data
        self.mask_tr, self.mask_va, self.mask_te = self.splits(data, seed, train_ratio, val_ratio)

    @property
    def processed_dir(self):
        return os.path.join(DATASET_FOLDER, self.__class__.__name__)

    @property
    def processed_file_names(self):
        return "data_processed.pt"

    def splits(self, data, seed, train_ratio, val_ratio):
        paths = [
            os.path.join(
                self.processed_dir,
                f"mask_seed{seed}_train{str(train_ratio).replace('.', 'p')}_val{str(val_ratio).replace('.', 'p')}_{name}.npy",
            )
            for name in ["tr", "va", "te"]
        ]
        if any(not os.path.exists(path) for path in paths):
            for path, mask in zip(paths, utils.get_masks(data.y, seed, train_ratio, val_ratio)):
                np.save(path, mask)

        return [np.load(path) for path in paths]

    def process(self):
        """
        Called if any of self.processed_file_names don't exist in folder self.processed_dir
        """

        # load data
        inputs = utils.open_pm25()
        inputs = inputs.reshape(-1, inputs.shape[-1])  # flatten in space, keep time (i.e. last dimension)
        outputs = utils.open_dem().flatten()  # the DEM

        # don't use pixels in graph that don't have inputs or where the output is nan (i.e. over sea)
        mask = ~np.all(np.isnan(inputs), axis=1) * ~np.isnan(outputs)
        outputs = outputs[mask]
        inputs = inputs[mask]

        # normalize inputs
        ip_min, ip_max = np.nanmin(inputs), np.nanmax(inputs)
        inputs = (inputs - ip_min) / (ip_max - ip_min)
        # normalize outputs
        op_min, op_max = np.nanmin(outputs), np.nanmax(outputs)
        outputs = (outputs - op_min) / (op_max - op_min)

        if outputs.ndim == 1:
            outputs = outputs[:, None]  # make shape (n, 1) instead of (n,)

        # the 'pos' attribute is used by pre_transform=T.KNNGraph(k=10, force_undirected=True) to create undirected graph
        data_list = [
            Data(
                pos=torch.tensor(inputs, dtype=torch.float),
                x=torch.tensor(inputs, dtype=torch.float),
                y=torch.tensor(outputs, dtype=torch.float),
            )
        ]

        # from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html#creating-in-memory-datasets
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # saving a huge Python list is slow so it gets collated into one huge Data object via torch_geometric.data.InMemoryDataset.collate()
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def read(self):
        data = torch.load(os.path.join(self.path, "data.pt"))
        return data
