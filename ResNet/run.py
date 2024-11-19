import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from download_dataset import *
from ResNet import *
from cuda_check import *

def evaluate_accuracy(data_loader, model, device):
    model.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            acc_sum += (output.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return acc_sum / n

def main():
    device, n = check_cuda_support()
    time_start = time.time()
    model = ResNet(3,100)

    selected_gpu = select_gpu_with_most_memory(device, n)
    if selected_gpu is not None:
        device = torch.device(f'cuda:{selected_gpu}')

    model.to(device)

    data_dir = r'/nfs/u20/zhoul83/files/ImageNet'
    print(f"Path: {data_dir} exists: {os.path.exists(data_dir)}")
    train_loader, test_loader = get_data_loaders(data_dir)

    # Check GPU availability and adjust model
    model = check_parallel_availability(device, n, model, train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_epochs = 10
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(n_epochs):
        model.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * data.size(0)
            train_acc_sum += (output.argmax(dim=1) == target).sum().item()
            n += data.size(0)

            if batch_idx % 16 == 0:
                print(f'\nEpoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        train_loss, train_acc = train_loss_sum / n, train_acc_sum / n
        test_acc = evaluate_accuracy(test_loader, model, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"\nEpoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    print(f"Training time: {time.time() - time_start:.2f}s")

    # Plot training progress
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), train_accs, label='Train Acc')
    plt.plot(range(1, n_epochs + 1), test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Visualization of predictions after training
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        predictions = output.argmax(dim=1)

    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(64):
        image = data[i].cpu().numpy().transpose((1, 2, 0))
        axes[i].imshow(image)
        axes[i].set_title(f'Pred: {predictions[i].item()}, True: {target[i].item()}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
