
#https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet
import torch.optim as optim
import torch.nn as nn
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])#입력 이미지를 전처리 해주는 함수
#cifar 10 데이터 셋 Load
classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
if __name__ == "__main__":
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AlexNet_model =AlexNet(num_classes=10)
    AlexNet_model.to(device)
    print(AlexNet_model.eval())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(AlexNet_model.parameters(), lr=0.001)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)            
            optimizer.zero_grad()            
            output = AlexNet_model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()            
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = AlexNet_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = AlexNet_model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

        avg = 0
    for i in range(10):
        temp = (100 * class_correct[i] / class_total[i])
    avg = avg + temp
    avg = avg/10
    print('Average accuracy = ', avg)

