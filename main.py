import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

ROOT = ".data"

train_data = datasets.MNIST(root=ROOT, train=True, download=True)

mean = train_data.data.float().mean() / 255
std = train_data.data.float().std() / 255

print(f"Calculated mean: {mean}")
print(f"Calculated std: {std}")

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(5, fill=(0,)),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ]
)

test_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
)

train_data = datasets.MNIST(
    root=ROOT, train=True, download=True, transform=train_transforms
)

test_data = datasets.MNIST(
    root=ROOT, train=False, download=True, transform=test_transforms
)

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(
    train_data, [n_train_examples, n_valid_examples]
)

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

BATCH_SIZE = 64

train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data, batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE)


def plot_filter(images, filter):

    images = images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    filter = torch.FloatTensor(filter).unsqueeze(0).unsqueeze(0).cpu()

    n_images = images.shape[0]

    filtered_images = F.conv2d(images, filter)

    fig = plt.figure(figsize=(20, 5))

    for i in range(n_images):

        ax = fig.add_subplot(2, n_images, i + 1)
        ax.imshow(images[i].squeeze(0), cmap="bone")
        ax.set_title("Original")
        ax.axis("off")

        image = filtered_images[i].squeeze(0)

        ax = fig.add_subplot(2, n_images, n_images + i + 1)
        ax.imshow(image, cmap="bone")
        ax.set_title("Filtered")
        ax.axis("off")


N_IMAGES = 5

images = [image for image, label in [test_data[i] for i in range(N_IMAGES)]]

horizontal_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

plot_filter(images, horizontal_filter)

horizontal_filter = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

plot_filter(images, horizontal_filter)

vertical_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

plot_filter(images, vertical_filter)

vertical_filter = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]

plot_filter(images, vertical_filter)

diagonal_filter = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]

plot_filter(images, diagonal_filter)


def plot_subsample(images, pool_type, pool_size):

    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()

    if pool_type.lower() == "max":
        pool = F.max_pool2d
    elif pool_type.lower() in ["mean", "avg"]:
        pool = F.avg_pool2d
    else:
        raise ValueError(f"pool_type must be either max or mean, got: {pool_type}")

    n_images = images.shape[0]

    pooled_images = pool(images, kernel_size=pool_size)

    fig = plt.figure(figsize=(20, 5))

    for i in range(n_images):

        ax = fig.add_subplot(2, n_images, i + 1)
        ax.imshow(images[i].squeeze(0), cmap="bone")
        ax.set_title("Original")
        ax.axis("off")

        image = pooled_images[i].squeeze(0)

        ax = fig.add_subplot(2, n_images, n_images + i + 1)
        ax.imshow(image, cmap="bone")
        ax.set_title("Subsampled")
        ax.axis("off")


plot_subsample(images, "max", 2)

plot_subsample(images, "max", 3)

plot_subsample(images, "avg", 2)

plot_subsample(images, "avg", 3)


class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):

        # x = [batch size, 1, 28, 28]

        x = self.conv1(x)

        # x = [batch size, 6, 24, 24]

        x = F.max_pool2d(x, kernel_size=2)

        # x = [batch size, 6, 12, 12]

        x = F.relu(x)

        x = self.conv2(x)

        # x = [batch size, 16, 8, 8]

        x = F.max_pool2d(x, kernel_size=2)

        # x = [batch size, 16, 4, 4]

        x = F.relu(x)

        x = x.view(x.shape[0], -1)

        # x = [batch size, 16*4*4 = 256]

        h = x

        x = self.fc_1(x)

        # x = [batch size, 120]

        x = F.relu(x)

        x = self.fc_2(x)

        # x = batch size, 84]

        x = F.relu(x)

        x = self.fc_3(x)

        # x = [batch size, output dim]

        return x, h


OUTPUT_DIM = 10

model = LeNet(OUTPUT_DIM)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for x, y in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for x, y in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


EPOCHS = 20

best_valid_loss = float("inf")

for epoch in trange(EPOCHS, desc="Epochs"):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut2-model.pt")

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

model.load_state_dict(torch.load("tut2-model.pt"))

test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")


def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for x, y in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


images, labels, probs = get_predictions(model, test_iterator, device)

pred_labels = torch.argmax(probs, 1)


def plot_confusion_matrix(labels, pred_labels):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=range(10))
    cm.plot(values_format="d", cmap="Blues", ax=ax)


plot_confusion_matrix(labels, pred_labels)

corrects = torch.eq(labels, pred_labels)

incorrect_examples = []

for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image, label, prob))

incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)


def plot_most_incorrect(incorrect, n_images):

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        ax.imshow(image.view(28, 28).cpu().numpy(), cmap="bone")
        ax.set_title(
            f"true label: {true_label} ({true_prob:.3f})\n"
            f"pred label: {incorrect_label} ({incorrect_prob:.3f})"
        )
        ax.axis("off")
    fig.subplots_adjust(hspace=0.5)


N_IMAGES = 25

plot_most_incorrect(incorrect_examples, N_IMAGES)


def get_representations(model, iterator, device):

    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():

        for x, y in tqdm(iterator):

            x = x.to(device)

            y_pred, h = model(x)

            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels


outputs, intermediates, labels = get_representations(model, train_iterator, device)


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def plot_representations(data, labels, n_images=None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10")
    handles, labels = scatter.legend_elements()
    ax.legend(handles=handles, labels=labels)


output_pca_data = get_pca(outputs)
plot_representations(output_pca_data, labels)

intermediate_pca_data = get_pca(intermediates)
plot_representations(intermediate_pca_data, labels)


def get_tsne(data, n_components=2, n_images=None):
    if n_images is not None:
        data = data[:n_images]
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


N_IMAGES = 5_000

output_tsne_data = get_tsne(outputs, n_images=N_IMAGES)
plot_representations(output_tsne_data, labels, n_images=N_IMAGES)

intermediate_tsne_data = get_tsne(intermediates, n_images=N_IMAGES)
plot_representations(intermediate_tsne_data, labels, n_images=N_IMAGES)


def imagine_digit(model, digit, device, n_iterations=50_000):

    model.eval()

    best_prob = 0
    best_image = None

    with torch.no_grad():

        for _ in trange(n_iterations):

            x = torch.randn(32, 1, 28, 28).to(device)

            y_pred, _ = model(x)

            preds = F.softmax(y_pred, dim=-1)

            _best_prob, index = torch.max(preds[:, digit], dim=0)

            if _best_prob > best_prob:
                best_prob = _best_prob
                best_image = x[index]

    return best_image, best_prob


DIGIT = 3

best_image, best_prob = imagine_digit(model, DIGIT, device)

print(f"Best image probability: {best_prob.item()*100:.2f}%")

plt.imshow(best_image.squeeze(0).cpu().numpy(), cmap="bone")
plt.axis("off")


def plot_filtered_images(images, filters):

    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    filters = filters.cpu()

    n_images = images.shape[0]
    n_filters = filters.shape[0]

    filtered_images = F.conv2d(images, filters)

    fig = plt.figure(figsize=(20, 10))

    for i in range(n_images):

        ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters))
        ax.imshow(images[i].squeeze(0), cmap="bone")
        ax.set_title("Original")
        ax.axis("off")

        for j in range(n_filters):
            image = filtered_images[i][j]
            ax = fig.add_subplot(
                n_images, n_filters + 1, i + 1 + (i * n_filters) + j + 1
            )
            ax.imshow(image.numpy(), cmap="bone")
            ax.set_title(f"Filter {j+1}")
            ax.axis("off")


N_IMAGES = 5

images = [image for image, label in [test_data[i] for i in range(N_IMAGES)]]
filters = model.conv1.weight.data

plot_filtered_images(images, filters)

plot_filtered_images([best_image], filters)


def plot_filters(filters):

    filters = filters.cpu()

    n_filters = filters.shape[0]

    fig = plt.figure(figsize=(20, 10))

    for i in range(n_filters):

        ax = fig.add_subplot(1, n_filters, i + 1)
        ax.imshow(filters[i].squeeze(0), cmap="bone")
        ax.axis("off")


plot_filters(filters)
