import torch
import os
from torchvision.transforms import transforms
from Dataset_Generation import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from model import Net

base_dir = os.getcwd()

normalization_factors = torch.load('normalization_factors_object_raw_images.pt')

transform_test = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(normalization_factors['img_mean'], normalization_factors['img_std'])
])

# transform_test_grayscale = transforms.Compose([
#     transform_test,
#     transforms.Grayscale(),
#     transforms.Normalize(normalization_factors['grayscale_img_mean'], normalization_factors['grayscale_img_std'])
# ])

@torch.no_grad()
def evaluation(model, dataLoader, device, img_show=False):
    correct = [0, ] * len(object_types)
    total = [0, ] * len(object_types)
    accuracy = [0., ]  * len(object_types)
    model.eval()
    denormalization_factors = torch.load('denormalization_factors_object_raw_images.pt')
    denorm = transforms.Normalize(denormalization_factors['denorm_mean'], denormalization_factors['denorm_std'])
    with torch.no_grad():
        for data in dataLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(object_types)):
                total[i] += (labels == i).sum().item()
                correct[i] += np.logical_and((predicted == labels).cpu().numpy(), (labels == i).cpu().numpy()).sum().item()
            if img_show and (predicted != labels).sum().item() > 1: # subject to change, should work for now but ignores batches with only 1 wrong pic
                wrong_idx = np.where((predicted != labels).cpu() != 0)[0].reshape(-1, ).tolist()
                # import pdb; pdb.set_trace()
                for idx in wrong_idx:
                    example = np.transpose(denorm(images[idx]).cpu().numpy(), (1, 2, 0)) # use this if input images are normalized
                    # example = np.transpose(images[idx].cpu().numpy(), (1, 2, 0))
                    plt.imshow(example.astype(np.uint8))
                    plt.title("predicted {}, should be {}".format(object_types[predicted[idx]], object_types[labels[idx]]))
                    plt.show()
    all_total = 0
    all_correct = 0
    for i in range(len(object_types)):
        all_total += total[i]
        all_correct += correct[i]
    print("Evaluated on all {} images...".format(all_total))
    for i in range(len(object_types)):
        accuracy[i] = 100 * correct[i] / total[i]
        print('Accuracy of the network on object {} from {} test images is: {}%'.format(object_types[i], total[i], accuracy[i]))
    print('Average accuracy of the network on test images is: {}%'.format(100 * all_correct / all_total))
    return accuracy


if __name__ == "__main__":
    bsz = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(base_dir, "models", "model_resnet18_89_15_20220915001621.pth"))
    print("number of trained parameters: %d" %
          (sum([param.nelement() for param in model.parameters() if param.requires_grad])))
    print("number of total parameters: %d" % (sum([param.nelement() for param in model.parameters()])))
    try:
        test_set = TouchSensorObjectsDataset(csv_file_path='object_raw_images_test.csv', transform=transform_test)
    except Exception as e:
        dataset = TouchSensorObjectsDataset(dataset_describer_path, transform=transform_test)
        _, test_set = torch.utils.data.random_split(dataset, [1800, 700])
        print("can't load test set because {}, load random subset now".format(e))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bsz, shuffle=False, num_workers=2)

    evaluation(model, testloader, device, img_show=True)
