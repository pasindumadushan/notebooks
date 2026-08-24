import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0
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


def efficientnet_lite_data_modeling(
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

    # Augmented transform for training — increases effective dataset size
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Clean transform for test — no augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset   = ImageFolder(
        root=os.path.join(processed_dir, "val"),
        transform=eval_transform
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -----------------------------------------
    # Model: EfficientNet-B0 (lightweight baseline)
    # -----------------------------------------

    model = efficientnet_b0(weights="DEFAULT")

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # -----------------------------------------
    # Training
    # -----------------------------------------

    print("=" * 60)
    print("TRAINING: EfficientNet-B0 Lite Baseline")
    print("=" * 60)
    print(f"\nDevice        : {device}")
    print(f"Train Samples : {len(train_dataset)}")
    print(f"Val Samples   : {len(val_dataset)}")
    print(f"Classes       : {train_dataset.classes}")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {learning_rate}")
    print()

    log_records      = []
    best_val_loss    = float("inf")
    best_model_state = None
    patience         = 5
    patience_counter = 0

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

        # -----------------------------------------
        # Validation
        # -----------------------------------------

        model.eval()

        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                labels  = labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels, num_classes)
                val_loss    += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        val_loss_avg = val_loss / val_total
        val_acc      = val_correct / val_total

        log_records.append({
            "epoch"       : epoch,
            "loss"        : round(epoch_loss, 4),
            "accuracy"    : round(epoch_acc, 4),
            "val_loss"    : round(val_loss_avg, 4),
            "val_accuracy": round(val_acc, 4),
        })

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:>3}/{epochs}]"
            f"  Loss: {epoch_loss:.4f}"
            f"  Acc: {epoch_acc:.4f}"
            f"  Val Loss: {val_loss_avg:.4f}"
            f"  Val Acc: {val_acc:.4f}"
            f"  LR: {current_lr:.2e}"
        )

        # Save best model; stop early if val loss stops improving
        if val_loss_avg < best_val_loss:
            best_val_loss    = val_loss_avg
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    # -----------------------------------------
    # Save model & training report
    # -----------------------------------------

    model_path = os.path.join(processed_dir, "efficientnet_lite_baseline.pth")

    # Restore best weights (lowest val loss) before saving and testing
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    torch.save(model.state_dict(), model_path)

    report_df = pd.DataFrame(log_records)

    report_df.to_csv("efficientnet_lite_modeling_report.csv", index=False)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel Saved   : {model_path}")
    print(f"Report Saved  : efficientnet_lite_modeling_report.csv")

    # -----------------------------------------
    # Test Evaluation
    # -----------------------------------------

    test_dir = os.path.join(processed_dir, "test")

    test_dataset = ImageFolder(
        root=test_dir,
        transform=eval_transform
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

    test_report_df.to_csv("efficientnet_lite_test_report.csv", index=False)

    print()
    print(f"Test Report   : efficientnet_lite_test_report.csv")

    return report_df, test_report_df
