from download_dataset import *
from neural_network import *
from cuda_check import *
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from time import time

def main():
    device = check_cuda_support()
    time_start = time()
    model = NN()
    model.to(device)

    train_loader, test_loader = get_data_loaders()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    n_epochs = 10

    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

    print(f"Training time: {time() - time_start}s")

    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        predictions = output.argmax(dim=1)

    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(64):
        image = data[i].cpu().numpy().transpose((1, 2, 0))
        axes[i].imshow(image)
        axes[i].set_title(f'Pred: {predictions[i].item()}, True: {target[i].item()}')
        axes[i].axis('off')

    plt.title("Batch results")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()