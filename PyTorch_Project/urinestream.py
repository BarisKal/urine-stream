import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.basiccnn_model import BasicCNN
from models.complexcnn_model import ComplexCNN
from utils.util import calculate_statistics, save_pandas_df_to_csv, parse_configuration
from datasets.dataloader import CustomDatasetDataLoader
from visualizations.visualize import plot_loss
from models.cnn_classifier import Model


def main(config: dict):
    """
    The main function reads the configuration file (configfile.json) passed as argument
    and runs the training and validation loop
    """
    model_name = config['model_params']['model_name']

    print('Initializing train and validation datasets for model {0}'.format(model_name))

    # Means and Standard Deviations per image channel can be passed through the configuration file for normalizing images.
    # If not passed, they will be calculated here
    if(config['calculate_stats_on_testset']):
        print('Calculating statistics of test dataset for normalizing datasets')
        config_train = config['train_dataset_params']
        rgb_mean, rgb_std = calculate_statistics(config_train['dataset_path'],
                                                 config_train['dataset_label_path'],
                                                 config_train['delimiter'],
                                                 True,
                                                 config['number_channels'])

        print('RGB Mean values = {0}'.format(rgb_mean))
        print('RGB STD values = {0}'.format(rgb_std))

        train_dataset_loader = CustomDatasetDataLoader(
            config['train_dataset_params'], rgb_mean, rgb_std)
        valid_dataset_loader = CustomDatasetDataLoader(
            config['validation_dataset_params'], rgb_mean, rgb_std)
        test_dataset_loader = CustomDatasetDataLoader(
            config['test_dataset_params'], rgb_mean, rgb_std)
    else:
        print('Taking normalization factors from config file')
        train_dataset_loader = CustomDatasetDataLoader(
            config['train_dataset_params'])
        valid_dataset_loader = CustomDatasetDataLoader(
            config['validation_dataset_params'])
        test_dataset_loader = CustomDatasetDataLoader(
            config['test_dataset_params'])

    print('The number of training samples = {0}'.format(len(train_dataset_loader)))
    print('The number of validation samples = {0}'.format(len(valid_dataset_loader)))
    print('The number of test samples = {0}'.format(len(test_dataset_loader)))

    if(model_name == 'complex_cnn'):
        model = ComplexCNN(config['model_params']['use_dropout'],
                           config['model_params']['dropout_probability'])
    else:
        model = BasicCNN()

    cnn_model = Model(model,
                      nn.CrossEntropyLoss(reduction='mean'),
                      optim.SGD(model.parameters(
                      ), lr=config['model_params']['lr'], weight_decay=config['model_params']['weight_decay']),
                      train_dataset_loader.dataloader,
                      valid_dataset_loader.dataloader)

    # Use if you want to convert model to ONNX
    #if(cnn_model.train_on_gpu):
            #res = config['test_dataset_params']['dataset_resolution']
            #dummy_input_onnx = torch.randn(1, 3, 96, 96, device='cuda')

    # Main Train/Validation Loop
    non_improving_epochs = 0
    cnn_model.train_losses = []
    cnn_model.validation_losses = []
    epochs = config['model_params']['max_epochs']

    minimum_validation_loss = np.inf
    for epoch in range(epochs):
        train_loss = cnn_model.train()
        cnn_model.train_losses.append(train_loss)

        validation_loss = cnn_model.validate()
        cnn_model.validation_losses.append(validation_loss)

        print('Epoch: {} \tTraining Loss: {:.7f} \tValidation Loss: {:.7f}'.format(
            epoch + 1, train_loss, validation_loss))
        if validation_loss < minimum_validation_loss:
            print('Validation loss decreased ({:.7f} --> {:.7f}).  Saving model ...'.format(
                minimum_validation_loss, validation_loss))
            torch.save(cnn_model.model.state_dict(), config['model_params']['best_path'] + model_name + '_best_weights.pt')

            # Use if you want to convert model to ONNX
            #if(cnn_model.train_on_gpu):
                #print('Save ONNX-Model')
                #torch.onnx.export(cnn_model.model.eval(), dummy_input_onnx,
                                 #config['model_params']['best_path'] + model_name + '_best_weights.onnx',
                                 #input_names=["u_image"], output_names=["stream_shape"], verbose=False)

            minimum_validation_loss = validation_loss
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
            if non_improving_epochs > config['model_params']['early_stopping']:
                print('Early stopping ... \nNo improvement since {0} epochs'.format(
                    non_improving_epochs))
                break

    # Save last weights
    print('Saving last weights to {0}'.format(config['model_params']['last_path']))

    # Use if you want to convert model to ONNX
    #if(cnn_model.train_on_gpu):
        #torch.onnx.export(cnn_model.model.eval(), dummy_input_onnx,
                          #config['model_params']['last_path'] + model_name + '_last_weights.onnx', 
                          #input_names=["u_image"], output_names=["stream_shape"], verbose=True)

    torch.save(cnn_model.model.state_dict(), config['model_params']['last_path'] + model_name + '_last_weights.pt')


    test_accuracy, predictions = cnn_model.predict(
        test_dataset_loader, config['model_params']['best_path'] + model_name + '_best_weights.pt')
    save_pandas_df_to_csv(predictions, './' + model_name + '_predictions.csv', True, ';')

    # Save plot with training and validation loss
    plot_loss(cnn_model.train_losses, cnn_model.validation_losses, test_accuracy,
              './visualizations/plots/' + model_name + '_losses_urinestream.png', 'epochs', 'losses')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='Path to the configfile')

    print('Reading config file')
    args = parser.parse_args()
    main(parse_configuration(args.configfile))
