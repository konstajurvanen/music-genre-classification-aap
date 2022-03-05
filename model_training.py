import torch
import os
import torch.nn as nn
import numpy as np

from copy import deepcopy
from torch.utils.data import DataLoader
from genre_clf import GenreClf
from dataset import GenreDataset


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    dir = './genres/'
    # Access label names indexing with argmax of one-hot
    labels = [f.name for f in os.scandir(dir) if f.is_dir()]

    # Define hyper-parameters

    epochs = 50
    learning_rate = 1e-4
    n_mels = 40
    batch_size = 1
    model_params = {
        'cnn_channels_out_1': 1,
        'cnn_kernel_1': 1,
        'cnn_stride_1': 1,
        'cnn_padding_1': 0,
        'pooling_kernel_1': 1,
        'pooling_stride_1': 1,
        'cnn_channels_out_2': 1,
        'cnn_kernel_2': 1,
        'cnn_stride_2': 1,
        'cnn_padding_2': 0,
        'pooling_kernel_2': 1,
        'pooling_stride_2': 1,
        'cnn_channels_out_3': 1,
        'cnn_kernel_3': 1,
        'cnn_stride_3': 1,
        'cnn_padding_3': 0,
        'pooling_kernel_3': 1,
        'pooling_stride_3': 1,
        'fc_out_1': 10,
        'fc_out_2': 10,
        'clf_output_classes': len(labels),
        'dropout_conv_1': 0.1,
        'dropout_conv_2': 0.1,
        'dropout_conv_3': 0.1,
        'dropout_fc_1': 0.1,
        'dropout_fc_2': 0.1}

    # Load datasets and generate iterators

    training_iterator = DataLoader(dataset=GenreDataset(data_dir='training', n_mels=n_mels),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=False)

    test_dataset = GenreDataset(data_dir='testing', n_mels=n_mels)
    test_iterator = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False)

    # Instantiate model

    model = GenreClf(**model_params)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    lowest_training_loss = 1e10
    best_training_epoch = 0
    patience = 3
    patience_counter = 0

    best_model = None

    for epoch in range(epochs):
        epoch_loss_train = []

        model.train()

        for i, batch in enumerate(training_iterator):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            lbl = torch.argmax(y)
            print(f'label: {labels[lbl]} has shape: {x.shape}')
            y_hat = model(x)
            loss = loss_function(input=y_hat, target=y)
            loss.backward()
            optimizer.step()
            epoch_loss_train.append(loss.item())

        epoch_loss_train = np.array(epoch_loss_train).mean()

        print(f'Epoch {epoch} training loss : {epoch_loss_train:.4f}')

        if epoch_loss_train < lowest_training_loss:
            lowest_training_loss = epoch_loss_train
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_training_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= patience or epoch == epochs-1:
            print('\nExiting training', end='\n\n')
            print(f'Best epoch {best_training_epoch} with training loss {lowest_training_loss}', end='\n\n')

            if best_model is None:
                print('No best model.')
            else:
                print('Starting testing', end=' | ')
                testing_loss = []
                model.eval()
                y_pred = torch.zeros(len(test_dataset), len(labels))
                y_true = torch.zeros(len(test_dataset), len(labels))

                with torch.no_grad():
                    for i, batch in enumerate(test_iterator):
                        x, y = batch
                        x = x.to(device=device, dtype=torch.float)
                        y = y.to(device=device, dtype=torch.float)
                        y_true[i * batch_size:i * batch_size + batch_size, :] = y
                        y_hat = model(x)
                        loss = loss_function(input=y_hat, target=y)
                        y_pred[i * batch_size:i * batch_size + batch_size, :] = y_hat
                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:.4f}')
                break


if __name__ == '__main__':
    main()