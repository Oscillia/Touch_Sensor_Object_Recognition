import torch
import os
from torchvision.transforms import transforms
from Dataset_Generation import *
# from model import Net

base_dir = os.path.dirname(__file__)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])])


@torch.no_grad()
def evaluation(model, dataLoader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the images: %d %%' % (accuracy))
    return accuracy


if __name__ == "__main__":
    bsz = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(base_dir, "models", "model_resnet18_0.pth"))
    print("number of trained parameters: %d" %
          (sum([param.nelement() for param in model.parameters() if param.requires_grad])))
    print("number of total parameters: %d" % (sum([param.nelement() for param in model.parameters()])))
    try:
        test_set = TouchSensorObjectsDataset(root=base_dir, split='test', transform=transform)
    except Exception as e:
        dataset = TouchSensorObjectsDataset(dataset_describer_path, root_dir=os.path.join(base_dir, "dataset"), transform=transform)
        _, test_set = torch.utils.data.random_split(dataset, [1600, 500])
        print("can't load test set because {}, load random subset now".format(e))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bsz, shuffle=False, num_workers=2)

    evaluation(model, testloader, device)
