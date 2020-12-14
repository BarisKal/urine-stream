# util.py
import os
import json
import csv
import cv2
import numpy as np

def parse_configuration(config_file_path: str):
    """Loads config file if a string was passed and return the input if a dictionary was passed.
    """
    if isinstance(config_file_path, str):
        with open(config_file_path) as json_file:
            return json.load(json_file)
    else:
        return config_file_path


def calculate_statistics(path_to_data: str, path_to_labels: str, delim: chr, skip_header: bool, num_channels: int):
    """Calculate mean and std statistics of input images for normalizing datasest
    """    
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)
    pixel_num = 0 # store all pixel number in the dataset
    filenames = read_csv_to_list(path_to_labels, delim, skip_header)

    for row in filenames:
        img_path = os.path.join(path_to_data, row[0])
        try:
            img = cv2.imread(img_path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
            img = img / 255.0
            pixel_num += (img.size / num_channels)
            channel_sum += np.sum(img, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(img), axis=(0, 1))
        except:
            print('Image {0} couldn\'t processed during calculating statistics.'.format(img_path))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
        
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
        
    return rgb_mean, rgb_std

def read_csv_to_list(path_to_data: str, delim: chr, skip_header: bool) -> list:
    """Read CSV file and return as list
    """
    with open(path_to_data, 'r') as file:
        reader = csv.reader(file, delimiter = delim)
        if(skip_header):
            next(reader, None)  # skip the header

        return_list = list(reader)
    
    return return_list

def save_pandas_df_to_csv(pandas_df, path: str = './predictions_urinestream.csv', write_header: bool = True, delimeter: chr = ','):
    """Save pandas df as CSV
    """
    if(pandas_df is not None):
        pandas_df.to_csv(path, header = write_header, sep = delimeter, index=False)
        print('Pandas dataframe is written to {0}'.format(path))
    else:
        print('Can\'t write Pandas dataframe to file')