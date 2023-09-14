import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

from linear_cut import LinearCut


batch_size = 2
in_features = 4
out_features = 8

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        # self.fc1   = nn.Linear(1024, 1024)
        # self.fc2   = nn.Linear(1024, 1024)

        self.fc1   = LinearCut(in_features, out_features)
        # self.fc2   = LinearCut(out_features, out_features)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.fc2(out)

        return out
    


def train(net,criterion,optimizer,trainloader,device):

    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # for batch_idx, (data, targets) in enumerate(trainloader):
    for batch_idx in range(10):
        data = torch.randint(0, 7, (batch_size, in_features), dtype=torch.float32)
        targets = torch.randint(0, 8, (batch_size, ), dtype=torch.float32)

        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, targets)
        # loss = outputs.sum()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = train_loss/(batch_idx+1)
    train_acc = 100.0*(correct/total)
    print("Training loss is {} and training accuracy is {}".format(train_loss,train_acc))

def test(net,criterion,testloader,device):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        # for batch_idx, (data, targets) in enumerate(testloader):
        for batch_idx in range(5):
            data = torch.randint(0, 7, (batch_size, in_features), dtype=torch.float32)
            targets = torch.randint(0, 8, (batch_size, ), dtype=torch.float32)

            data, targets = data.to(device), targets.to(device)

            outputs = net(data)
            # print(outputs[1, :])
            # print(outputs.shape)
    #         loss = criterion(outputs, targets)

    #         test_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()

    # test_loss = test_loss/(batch_idx+1)
    # test_acc = 100.0*(correct/total)
    test_acc = 0
    print("Testing loss is {} and testing accuracy is {}".format(test_loss,test_acc))

def main():

    epochs = 1
    batch_size = 128
    lr = 0.01
    milestones = [10,15]
    load_path = './data'
    save_path = './checkpoints'

    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    
    device = torch.device('cuda:0')

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root=load_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root=load_path, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    net = SimpleModel()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(1, epochs+1):
        print("=====================================================================")
        print("Epoch : {}".format(epoch))
        train(net,criterion,optimizer,trainloader,device)
        test(net,criterion,testloader,device)

        torch.save(net.state_dict(), os.path.join(save_path,'epoch_{}.pth'.format(epoch)))

        scheduler.step()

if __name__ == '__main__':
    main()