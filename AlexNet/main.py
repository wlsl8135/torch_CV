
#https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet
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