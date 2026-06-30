import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from models import *
from tqdm import tqdm


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("loss function not found")
    return optimizer


def train(model, data_loader, optimizer, epoch, tqdm_on=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    criterion = nn.CrossEntropyLoss()

    if tqdm_on:
        for inputs, labels in tqdm(data_loader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    else:
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



def test(model, data_loader, extra_class=0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if extra_class!=0:
                outputs = outputs[:, :-extra_class]
                
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(data_loader)
    train_acc = correct / total
    return train_loss, train_acc


@timer
def train_save_model(train_loader, test_loader, model_name, optim_name, learning_rate, num_epochs, path, description, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = max(train_loader.dataset.targets) + 1
    print(f"num_classes:{num_classes}")

    model = get_model(model_name, num_classes)
    model = model.to(device)
    print(f"Model {model_name} loaded on {device}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    best_acc = 0

    # Track loss/accuracy across training and testing.
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{seconds:.0f}"
    
    start_time = time.time()
    for epoch in range(num_epochs):
        train(model=model, data_loader=train_loader, optimizer=optimizer, epoch=epoch, device=device)

        train_loss, train_acc = test(model=model, data_loader=train_loader, device=device)
        print(f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.2%}")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss,  test_acc = test(model=model, data_loader=test_loader, device=device)
        print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2%}")
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), path / f"{description}.pth")

        # Step the LR scheduler at the end of each epoch.
        scheduler.step()

        # Save the loss/accuracy curves.
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.title('Accuracy Curve')

        plt.tight_layout()
        plt.savefig(path/f'{description}_training_curves.png')
        plt.close()
        now_time = time.time()
        token_time = now_time - start_time
        total_time = (now_time-start_time)/(epoch + 1)*num_epochs
        left_time = total_time - token_time
        print(f"Time taken: {format_time(token_time)}/{format_time(total_time)}, Left time: {format_time(left_time)}")
    best_ckpt = path / f"{description}.pth"
    print(f"Model saved at {best_ckpt}  (best test acc: {best_acc:.2%})")
    # Return the best checkpoint, not the last-epoch model
    best_state = torch.load(best_ckpt, map_location=device)
    if isinstance(best_state, dict):
        best_state = {k.replace('module.', ''): v for k, v in best_state.items()}
        model.load_state_dict(best_state)
    return model


@timer
def finetune_save_model(train_loader, test_loader, model, optim_name, learning_rate, num_epochs, path, description):
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate)

    best_acc = 0

    # Track loss/accuracy across training and testing.
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{seconds:.0f}"
    
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
        train(model=model, data_loader=train_loader, optimizer=optimizer, epoch=epoch, tqdm_on=False)

        train_loss, train_acc = test(model=model, data_loader=train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss,  test_acc = test(model=model, data_loader=test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), path / f"{description}.pth")

        # Save the loss/accuracy curves.
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1) 
        plt.legend()
        plt.title('Accuracy Curve')

        plt.tight_layout()
        plt.savefig(path/f'{description}_training_curves.png')
        plt.close()
        now_time = time.time()
        token_time = now_time - start_time
        total_time = (now_time-start_time)/(epoch + 1)*num_epochs
        left_time = total_time - token_time
        print(f"Time taken: {format_time(token_time)}/{format_time(total_time)}, Left time: {format_time(left_time)}")
    print(f"Model saved at { path/f'{description}.pth'}")

    return model


def test_each_classes(model, loader, num_classes=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    res = ''

    cnt = torch.zeros(num_classes).to(torch.int64).to(device)
    pred_cnt = torch.zeros(num_classes).to(torch.int64).to(device)
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            probabilities = F.softmax(output, dim=1)

            # Predicted label = argmax of the softmax probabilities.
            pred = probabilities.argmax(dim=1)
            correct = (pred == target).to(torch.int64)

            # Accumulate per-class sample counts and correct predictions.
            cnt.scatter_add_(0, target, torch.ones_like(target))
            pred_cnt.scatter_add_(0, target, correct)

    accuracy = pred_cnt / cnt
    for i in range(num_classes):
        res += f'class {i} acc: {accuracy[i]:.2%}\n'
    return res


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    # Computes accuracy and can optionally plot a confusion matrix.
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        if mode == 'pruned':
            batch_y_predict = batch_y_predict[:, 0:10]

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    # print()

    if print_perform and mode != 'backdoor' and mode != 'widen' and mode != 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
    if print_perform and mode == 'widen':
        class_name = data_loader.dataset.classes.append('extra class')
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=class_name, digits=4))
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'pruned':
        # print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.title('{} confusion matrix'.format(name), loc='center')
        plt.show()

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


if __name__ == "__main__":
    class identity_model:
        def __call__(self, x):
            return x
    model = identity_model()
    dataset = []
    i=75
    wrong_data = torch.cat([torch.zeros((100-i, 1)), torch.ones((100-i, 1))], dim=1)
    right_data = torch.cat([torch.ones((i, 1)), torch.zeros((i, 1))], dim=1)
    data = torch.cat([wrong_data, right_data], dim=0)
    label = torch.zeros((100,)).to(torch.int64)
    dataset = dataset + [(data, label)]
    print(test_each_classes(model, dataset, 2))