import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms


def ordinal_severity_loss(outputs, labels, num_classes=5):
    """
    Ordinal cross-entropy loss that penalizes predictions in proportion
    to their distance from the true severity level.
    Predicting severity 4 for a severity-0 image is penalized more
    than predicting severity 1.
    """
    severity = torch.arange(
        num_classes,
        dtype=torch.float32,
        device=outputs.device
    )

    # distance from true severity to each class: shape (batch, num_classes)
    labels_f = labels.float().unsqueeze(1)
    weights  = (severity - labels_f).abs() + 1  # +1 so correct class has weight 1

    log_probs    = torch.log_softmax(outputs, dim=1)
    weighted_nll = -(weights * log_probs).sum(dim=1)

    return weighted_nll.mean()


def data_modeling(
        processed_dir,
        num_classes=5,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        img_size=224,
):

    train_dir = os.path.join(processed_dir, "train")

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # -----------------------------------------
    # Model: MobileNetV3 Small (baseline)
    # -----------------------------------------

    model = mobilenet_v3_small(weights="DEFAULT")

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # -----------------------------------------
    # Loss & Optimizer
    # -----------------------------------------

    # Ordinal loss: penalizes severity misclassification by distance
    # e.g. predicting 4 for true 0 costs 5x more than predicting 1
    criterion = ordinal_severity_loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -----------------------------------------
    # Training
    # -----------------------------------------

    print("=" * 60)
    print("TRAINING: MobileNetV3 Small Baseline")
    print("=" * 60)
    print(f"\nDevice        : {device}")
    print(f"Train Samples : {len(train_dataset)}")
    print(f"Classes       : {train_dataset.classes}")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {learning_rate}")
    print()

    log_records = []

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels, num_classes)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total

        log_records.append({
            "epoch"   : epoch,
            "loss"    : round(epoch_loss, 4),
            "accuracy": round(epoch_acc, 4),
        })

        print(
            f"Epoch [{epoch:>3}/{epochs}]"
            f"  Loss: {epoch_loss:.4f}"
            f"  Accuracy: {epoch_acc:.4f}"
        )

    # -----------------------------------------
    # Save model & report
    # -----------------------------------------

    model_path = os.path.join(processed_dir, "mobilenetv3_baseline.pth")

    torch.save(model.state_dict(), model_path)

    report_df = pd.DataFrame(log_records)

    report_df.to_csv("modeling_report.csv", index=False)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel Saved   : {model_path}")
    print(f"Report Saved  : modeling_report.csv")

    return report_df
