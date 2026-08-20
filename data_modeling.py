import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms


def ordinal_severity_loss(outputs, labels, num_classes=5):
    """
    Combined loss:
      - CrossEntropy: strong gradient toward the correct class
      - Ordinal penalty: penalizes probability mass on classes
        far from the true severity level
    """

    # Primary signal — drives correct class prediction
    ce_loss = torch.nn.functional.cross_entropy(outputs, labels)

    # Ordinal penalty — penalizes far-away probability mass
    severity = torch.arange(
        num_classes,
        dtype=torch.float32,
        device=outputs.device
    )

    labels_f        = labels.float().unsqueeze(1)
    distances       = (severity - labels_f).abs()     # shape (batch, num_classes)

    probs           = torch.softmax(outputs, dim=1)
    ordinal_penalty = (distances * probs).sum(dim=1).mean()

    return ce_loss + ordinal_penalty


def data_modeling(
        processed_dir,
        num_classes,
        epochs,
        batch_size,
        learning_rate,
        img_size,
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

    # -----------------------------------------
    # Balanced sampler (handles class imbalance)
    # -----------------------------------------

    class_counts  = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    class_weights  = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
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
    # Save model & training report
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

    # -----------------------------------------
    # Test Evaluation
    # -----------------------------------------

    test_dir = os.path.join(processed_dir, "test")

    test_dataset = ImageFolder(
        root=test_dir,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    test_loss    = 0.0
    test_correct = 0
    test_total   = 0

    # Per-class correct/total for per-class accuracy
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs   = model(images)
            loss      = criterion(outputs, labels, num_classes)

            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            test_correct += (predicted == labels).sum().item()
            test_total   += labels.size(0)

            for label, pred in zip(labels, predicted):
                class_total[label.item()]   += 1
                class_correct[label.item()] += (pred == label).item()

    test_loss_avg = test_loss / test_total
    test_acc      = test_correct / test_total

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nTest Samples  : {test_total}")
    print(f"Test Loss     : {test_loss_avg:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print()
    print("Per-Class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            cls_acc = class_correct[i] / class_total[i]
            print(f"  Severity {i}  : {cls_acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    test_report_df = pd.DataFrame([{
        "test_loss"    : round(test_loss_avg, 4),
        "test_accuracy": round(test_acc, 4),
        **{
            f"severity_{i}_accuracy": round(
                class_correct[i] / class_total[i], 4
            ) if class_total[i] > 0 else None
            for i in range(num_classes)
        }
    }])

    test_report_df.to_csv("test_report.csv", index=False)

    print()
    print(f"Test Report   : test_report.csv")

    return report_df, test_report_df
