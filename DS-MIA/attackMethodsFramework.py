import sys
sys.dont_write_bytecode = True
import numpy as np
import Models as models
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in
           dataToClip]
    return np.array(res)

def train_attack_model_RNN_mul(
        num_class=2,
        dataset='',
        epochs=100,
        batch_size=100,
        learning_rate=0.01,
        l2_ratio=1e-7,
        n_hidden=50,
        model='rnn'
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x, test_x = dataset
    num_classes = num_class
    if batch_size > len(train_x):
        batch_size = len(train_x)
    print('Building model with {} training data, {} classes...'.format(len(train_x), num_classes))

    train_data = models.TrData(train_x)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=models.collate_fn
    )
    test_data = models.TrData(test_x)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=models.collate_fn
    )

    onetr = train_data[0]# Tensor shape = (T, D)
    onepoint_size = onetr.size(1)
    input_size = onepoint_size - 1
    hidden_size = 50
    num_layers = 1

    if model == 'rnn':
        print('Using an RNN based model for attack...')
        net = models.lstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            batch_size=batch_size
        )
        net = net.to(device)
    elif model == 'rnnAttention':
        print('Using an RNN with atention model for attack...')
        net = models.LSTM_Attention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            batch_size=batch_size
        )
        net = net.to(device)
    elif model == 'transformer':
        print('Using a Transformer encoder model for attack...')
        net = models.Transformer_Attack(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            batch_size=batch_size,
            nhead=10,
            dropout=0.1,
            pooling='mean'
        ).to(device)
    else:
        print('Using an error type for attack model...')
        #transformer

    net.train()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    learning_rate = 0.01
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    print(
        'model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(model, device,
                                                                                                         epochs,
                                                                                                         batch_size,
                                                                                                         learning_rate,
                                                                                                         l2_ratio))
    count = 1

    print('Training...')
    for epoch in range(epochs):
        running_loss = 0
        for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(train_loader):
            X_vector = X_vector.to(device)
            Y_vector = Y_vector.to(device)

            output, _ = net(X_vector, len_of_oneTr)
            output = output.squeeze(0)
            Y_vector = Y_vector.long()
            loss = criterion(output, Y_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if optimizer.param_groups[0]['lr'] > 0.0005:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))
    print("Training finished!")
    print('Testing...')
    pred_y = []
    pred_y_prob = []
    test_y = []
    hidden_outputs = []
    net.eval()
    if batch_size > len(test_x):
        batch_size = len(test_x)
    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output, hidden_output = net(X_vector, len_of_oneTr)
        output = output.squeeze(0)
        out_y = output.detach().cpu()
        pred_y.append(np.argmax(out_y,axis=1))
        #pred_y_prob.append(out_y[:, 1])
        out_y_softmax = torch.softmax(out_y, dim=1).numpy()
        pred_y_prob.append(out_y_softmax)
        test_y.append(Y_vector.detach().cpu())
        hidden_output = hidden_output.detach().cpu()
        hidden_output = np.squeeze(hidden_output)
        hidden_outputs.append(hidden_output)
    pred_y = np.concatenate(pred_y)
    pred_y_prob = np.concatenate(pred_y_prob)
    hidden_outputs = np.concatenate(hidden_outputs)
    test_y = np.concatenate(test_y)
    print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('More detailed results:')
    print(classification_report(test_y, pred_y))

    return test_y, pred_y_prob, hidden_outputs

def AttackingWithShadowTraining_RNN(X_train, X_test, epochs=50, batch_size=20,modelType='rnn',num_class=2):
    dataset = (X_train,X_test)
    l2_ratio = 0.0001

    targetY, pre_member_label, hidden_outputs = train_attack_model_RNN_mul(
            num_class=num_class,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.01,
            n_hidden=64,
            l2_ratio=l2_ratio,
            model=modelType
        )

    return targetY, pre_member_label, hidden_outputs


def softmax(x):
    shift = np.amax(x, axis=1)
    shift = shift.reshape(-1, 1)
    x = x - shift
    exp_values = np.exp(x)
    denominators = np.sum(np.exp(x), axis=1)
    softmax_values = (exp_values.T / denominators).T
    return softmax_values



