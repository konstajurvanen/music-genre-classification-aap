from copy import deepcopy
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from genre_clf import GenreClf
from dataset import GenreDataset
from plotting import plot_confusion_matrix

# Make experiment tracking manageable
torch.manual_seed(52)
torch.backends.cudnn.deterministic = True

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    dir = './genres/'
    # Access label names indexing with argmax of one-hot
    labels = [f.name for f in os.scandir(dir) if f.is_dir()]

    # Define training parameters
    config = {
        "epochs": 30,
        "learning_rate": 1e-4,
        "batch_size": 8
    }
    n_mels = 80
    model_params = {
        'cnn_channels_out_1': 32,
        'cnn_kernel_1': 3,
        'cnn_stride_1': 1,
        'cnn_padding_1': 0,
        'pooling_kernel_1': 2,
        'pooling_stride_1': 2,
        'cnn_channels_out_2': 32,
        'cnn_kernel_2': 3,
        'cnn_stride_2': 1,
        'cnn_padding_2': 0,
        'pooling_kernel_2': 2,
        'pooling_stride_2': 2,
        'cnn_channels_out_3': 64,
        'cnn_kernel_3': 3,
        'cnn_stride_3': 1,
        'cnn_padding_3': 0,
        'pooling_kernel_3': 2,
        'pooling_stride_3': 2,
        'fc_out_1': 128,
        'clf_output_classes': len(labels),
        'dropout_conv_1': 0.,
        'dropout_conv_2': 0.,
        'dropout_conv_3': 0.,
        'dropout_fc_1': 0.4,
    }

    # Load datasets and generate iterators

    training_dataset = GenreDataset(data_dir='training', n_mels=n_mels)
    training_iterator = DataLoader(dataset=training_dataset,
                                   batch_size=config['batch_size'],
                                   shuffle=True,
                                   drop_last=False)

    test_dataset = GenreDataset(data_dir='testing', n_mels=n_mels)
    test_iterator = DataLoader(dataset=test_dataset,
                               batch_size=config['batch_size'],
                               shuffle=False,
                               drop_last=False)

    # Instantiate model

    model = GenreClf(**model_params)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        epoch_loss_train = []
        epoch_acc_train = 0

        model.train()

        for i, (x, y) in enumerate(training_iterator):
            optimizer.zero_grad()
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            y_hat = model(x)
            loss = loss_function(input=y_hat, target=y)
            loss.backward()
            optimizer.step()
            epoch_loss_train.append(loss.item())
            epoch_acc_train += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

        epoch_loss_train = np.array(epoch_loss_train).mean()

        print(f"epoch {epoch} Training loss {epoch_loss_train:7.4f} acc {epoch_acc_train/len(training_dataset):7.4f}%")

    print('Starting testing', end=' | ')
    testing_loss = []
    epoch_acc_test = 0
    model.eval()
    y_pred = torch.zeros(len(test_dataset), len(labels))
    y_true = torch.zeros(len(test_dataset), len(labels))

    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            y_true[i * config['batch_size']:i * config['batch_size'] + config['batch_size'], :] = y
            y_hat = model(x)
            loss = loss_function(input=y_hat, target=y)
            y_pred[i * config['batch_size']:i * config['batch_size'] + config['batch_size'], :] = y_hat
            testing_loss.append(loss.item())
            epoch_acc_test += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            
    cm = confusion_matrix(y_true.argmax(dim=1), y_pred.argmax(dim=1))
    plot_confusion_matrix(cm, classes=labels)

    testing_loss = np.array(testing_loss).mean()
    print(f'Testing loss: {testing_loss:.4f} acc {epoch_acc_test/len(test_dataset):7.4f}%')

    torch.save(model.state_dict(), 'genre-classifier.pt')


if __name__ == '__main__':
    main()
# %%
