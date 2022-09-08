import torch

def print_accr(model, loader, epoch_i, device):
    correct = 0
    total = 0
    # wrongimage = []
    # wrongpred = []
    # wronglabel = []
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images after epoch %d is: %.3f%%' % (epoch_i+1, 100.0 * correct / total))
    return correct/total

