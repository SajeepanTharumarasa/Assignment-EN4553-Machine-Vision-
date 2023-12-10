import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet


def data_loader(dir="data", batch_size=32):
    # Define the transformation to be applied to the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Create the dataset
    train_data = OxfordIIITPet(
        root=dir + "/train", split="trainval", transform=transform, download=True
    )
    test_data = OxfordIIITPet(
        root=dir + "/test", split="test", transform=transform, download=True
    )

    # Create a data loader
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader


def find_embeddings(data_loader, device):
    # Load pre-trained ResNet-50 model
    resnet50 = models.resnet50(pretrained=True)

    # Remove the last classifier layer (fully connected layer)
    model = nn.Sequential(*list(resnet50.children())[:-1])
    model.eval()

    # Move the model to GPU if available
    model.to(device)

    # Extract ResNet-50 embeddings and labels
    embeddings, labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            images, batch_labels = batch
            # Move the input images and labels to GPU if available
            images, batch_labels = images.to(device), batch_labels.to(device)

            batch_embeddings = model(images)
            embeddings.append(batch_embeddings)
            labels.append(batch_labels)

    # Concatenate embeddings and labels
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    # Flatten the embeddings
    embeddings = embeddings.view(embeddings.size(0), -1)

    return embeddings.cpu().numpy(), labels.cpu().numpy()


def main():
    # Load data & create data loader
    train_data_loader, test_data_loader = data_loader()

    # Check if GPU is available and use it, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Embedding find start")
    X_train, y_train = find_embeddings(train_data_loader, device)
    print("Training embedding found successfully")

    X_test, y_test = find_embeddings(test_data_loader, device)
    print("Test embedding found successfully")

    # Create and train k-NN classifier
    k = 37
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = knn_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    main()
