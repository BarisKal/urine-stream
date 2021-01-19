import os
import cv2
from torch.utils import data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class CustomDatasetDataLoader():
    """Create custom data loaders and pass datasets and labels to them.
       Furthermore, transformations and normalizations will be applied to the datasets.
    """
    def __init__(self, configuration: dict, rgb_mean: np.array = None, rgb_std: np.array = None):
        self.configuration = configuration

        self.labels = pd.read_csv(self.configuration['dataset_label_path'], sep=self.configuration['delimiter'])

        if(('data_transforms' in configuration) and ((rgb_mean and rgb_std) is not None)):
            self.dataset = CustomDataset(dataframe = self.labels, data_dir = configuration['dataset_path'], transform = eval(configuration['data_transforms']))
            self.dataset.transform.transforms.append(transforms.Normalize(rgb_mean, rgb_std))
        elif(('data_transforms' in configuration) and ((rgb_mean or rgb_std) is None)):
            self.dataset = CustomDataset(dataframe = self.labels, data_dir = configuration['dataset_path'], transform = eval(configuration['data_transforms']))
        else:
            self.dataset = CustomDataset(dataframe = self.labels, data_dir = configuration['dataset_path'], transform = None)
        
        self.dataloader = data.DataLoader(self.dataset, **configuration['loader_params'])
        print('Dataset and dataloader for {0} set was created'.format(configuration['dataset_name'].upper()))

    def _load_data_labels(self, configuration) -> pd.DataFrame:
        return pd.read_csv(configuration['dataset_label_path'], sep=configuration['separator'])

    def __len__(self):
        """Return the number of data in the dataset.
        """
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data.
        """
        for img in self.dataloader:
            yield img

class CustomDataset():
    def __init__(self, dataframe, data_dir, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_name, label = self.dataframe.iloc[index]
        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path)

        if self.transform != None:
            image = self.transform(image)
        
        return image, label