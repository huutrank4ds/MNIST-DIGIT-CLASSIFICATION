from tensorflow import keras
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from model import MyModel
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall
import json

# Chuẩn bị dữ liệu huấn luyện
class MnistDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)

    @staticmethod
    def one_hot_encode(label):
      new_label = np.zeros(10)
      new_label[label] = 1
      return new_label

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, self.one_hot_encode(label)

    def __len__(self):
        return len(self.images)
    
# Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# Rescale ảnh về [0, 1] (ảnh xám có giá trị từ 0-255)
X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32) / 255.0
# Chia tập train thành train và validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_dataset = MnistDataset(X_train, y_train)
val_dataset = MnistDataset(X_val, y_val)
test_dataset = MnistDataset(X_test, y_test)

# Hàm huấn luyện mô hình
def train():
    global best_val_loss

    accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)
    precision_metric = Precision(task="multiclass", num_classes=10, average='macro').to(device)
    recall_metric = Recall(task="multiclass", num_classes=10, average='macro').to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}:"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, loss = model(images, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Cập nhật các metric
            accuracy_metric.update(outputs, torch.argmax(labels, dim=1))
            precision_metric.update(outputs, torch.argmax(labels, dim=1))
            recall_metric.update(outputs, torch.argmax(labels, dim=1))

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        tqdm.write(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Lưu các giá trị metric sau mỗi epoch
        train_accuracies.append(accuracy_metric.compute().item())
        train_precisions.append(precision_metric.compute().item())
        train_recalls.append(recall_metric.compute().item())

        # Reset các metric sau mỗi epoch
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        model.eval()
        total_val_loss = 0

        for images, labels in tqdm(val_dataloader, desc=f"Valid Epoch {epoch + 1}:"):
            images, labels = images.to(device), labels.to(device)  # Chuyển dữ liệu lên GPU
            outputs, loss = model(images, labels)
            total_val_loss += loss.item()
            # Cập nhật các metric
            accuracy_metric.update(outputs, torch.argmax(labels, dim=1))
            precision_metric.update(outputs, torch.argmax(labels, dim=1))
            recall_metric.update(outputs, torch.argmax(labels, dim=1))

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        tqdm.write(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}")

        # Lưu các giá trị metric sau mỗi epoch
        val_accuracies.append(accuracy_metric.compute().item())
        val_precisions.append(precision_metric.compute().item())
        val_recalls.append(recall_metric.compute().item())

        # Reset các metric sau mỗi epoch
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        # Lưu mô hình nếu loss trên tập validation giảm
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            tqdm.write(f"Best model saved with val loss: {best_val_loss:.4f}")

# Hàm đánh giá mô hình trên tập test
def evaluate():
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)
    precision_metric = Precision(task="multiclass", num_classes=10, average='macro').to(device)
    recall_metric = Recall(task="multiclass", num_classes=10, average='macro').to(device)

    total_test_loss = 0

    for images, labels in tqdm(test_dataloader, desc="Test:"):
        images, labels = images.to(device), labels.to(device)  # Chuyển dữ liệu lên GPU
        outputs, loss = model(images, labels)
        total_test_loss += loss.item()
        # Cập nhật các metric
        accuracy_metric.update(outputs, torch.argmax(labels, dim=1))
        precision_metric.update(outputs, torch.argmax(labels, dim=1))
        recall_metric.update(outputs, torch.argmax(labels, dim=1))

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = accuracy_metric.compute().item()
    test_precision = precision_metric.compute().item()
    test_recall = recall_metric.compute().item()

    tqdm.write(f"Test Loss: {avg_test_loss:.4f}")
    tqdm.write(f"Test Accuracy: {test_accuracy:.4f}")
    tqdm.write(f"Test Precision: {test_precision:.4f}")
    tqdm.write(f"Test Recall: {test_recall:.4f}")

    # Reset các metric sau khi đánh giá
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()


if __name__ == "__main__":
    # Tạo DataLoader cho các tập dữ liệu
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Khởi tạo các biến theo dõi quá trình huấn luyện
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []

    # Khởi tạo mô hình và các thành phần huấn luyện
    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Bắt đầu huấn luyện và đánh giá
    train()
    evaluate()

    # Lưu các giá trị loss và accuracy vào file json
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_precisions": train_precisions,
        "val_precisions": val_precisions,
        "train_recalls": train_recalls,
        "val_recalls": val_recalls
    }
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)
