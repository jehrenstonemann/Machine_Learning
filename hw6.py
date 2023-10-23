import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.
def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    if not training:
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    return loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        run_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            run_loss += loss.item() * 64
        acc = 100 * correct / total
        loss = run_loss / total
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total}({acc:.2f}%) Loss: {loss:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    INPUT: 
        model - the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    run_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            run_loss += loss.item() * 64
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        loss = run_loss / total
        print("Accuracy: " + str(acc) + "%")
        if show_loss:
            print("Average Loss: " + str(round(loss, 4)))


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle '
                                                                                                               'Boot']
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1) * 100
    values, indicies = torch.topk(prob, 3)
    for i in range(3):
        print(str(class_names[indicies[0][i]]) + ": " + str(round(values[0][i].item(), 2)) + "%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)
