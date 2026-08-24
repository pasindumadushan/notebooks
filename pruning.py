import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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

    ce_loss  = torch.nn.functional.cross_entropy(outputs, labels)

    severity = torch.arange(
        num_classes,
        dtype=torch.float32,
        device=outputs.device
    )

    labels_f        = labels.float().unsqueeze(1)
    distances       = (severity - labels_f).abs()
    probs           = torch.softmax(outputs, dim=1)
    ordinal_penalty = (distances * probs).sum(dim=1).mean()

    return ce_loss + ordinal_penalty


def _model_sparsity(model):
    """Return fraction of zero weights across all Conv2d and Linear layers."""
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.nelement()
            zeros += (module.weight == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def pruning(
        processed_dir,
        num_classes=5,
        epochs=10,
        batch_size=32,
        learning_rate=1e-5,
        img_size=224,
        pruning_amount=0.3,
):
    """
    Loads the distilled student model (MobileNetV3 Small), applies global
    L1 unstructured pruning, then fine-tunes to recover accuracy.

    pruning_amount : fraction of weights to prune across all Conv2d + Linear
                     layers (0.3 = remove the 30% of weights closest to zero)
    learning_rate  : keep this low (1e-5) — fine-tuning after pruning,
                     not training from scratch
    """

    train_dir   = os.path.join(processed_dir, "train")
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

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
    # Load distilled student model
    # -----------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_path = os.path.join(processed_dir, "student_distilled.pth")

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )
    model.load_state_dict(torch.load(student_path, map_location=device))
    model = model.to(device)

    # -----------------------------------------
    # Apply global L1 unstructured pruning
    # -----------------------------------------

    sparsity_before = _model_sparsity(model)

    # Collect all Conv2d and Linear weight tensors
    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    # Remove the lowest-magnitude pruning_amount fraction of weights globally
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount,
    )

    sparsity_after = _model_sparsity(model)

    # -----------------------------------------
    # Optimizer (low LR — fine-tuning, not retraining)
    # -----------------------------------------

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
    # Fine-tuning
    # -----------------------------------------

    print("=" * 60)
    print("PRUNING + FINE-TUNING: MobileNetV3 Small Student")
    print("=" * 60)
    print(f"\nDevice              : {device}")
    print(f"Train Samples       : {len(train_dataset)}")
    print(f"Val Samples         : {len(val_dataset)}")
    print(f"Classes             : {train_dataset.classes}")
    print(f"Epochs              : {epochs}")
    print(f"Batch Size          : {batch_size}")
    print(f"Learning Rate       : {learning_rate}")
    print(f"Pruning Amount      : {pruning_amount:.0%}")
    print(f"Sparsity Before     : {sparsity_before:.4f}")
    print(f"Sparsity After      : {sparsity_after:.4f}")
    print()

    log_records      = []
    best_val_loss    = float("inf")
    best_model_state = None
    patience         = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        correct      = 0
        total        = 0

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
    # Make pruning permanent, save model
    # -----------------------------------------

    # Restore best weights before making pruning permanent
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Remove pruning masks — bake zeroed weights into the actual parameters
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    final_sparsity = _model_sparsity(model)

    model_path = os.path.join(processed_dir, "student_pruned.pth")

    torch.save(model.state_dict(), model_path)

    report_df = pd.DataFrame(log_records)
    report_df.to_csv(os.path.join(reports_dir, "pruning_modeling_report.csv"), index=False)

    print()
    print("=" * 60)
    print("PRUNING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Sparsity  : {final_sparsity:.4f}")
    print(f"Model Saved     : {model_path}")
    print(f"Report Saved    : {os.path.join(reports_dir, 'pruning_modeling_report.csv')}")

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

    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels, num_classes)

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
        "pruning_amount": pruning_amount,
        "final_sparsity": round(final_sparsity, 4),
        "test_loss"     : round(test_loss_avg, 4),
        "test_accuracy" : round(test_acc, 4),
        **{
            f"severity_{i}_accuracy": round(
                class_correct[i] / class_total[i], 4
            ) if class_total[i] > 0 else None
            for i in range(num_classes)
        }
    }])

    test_report_df.to_csv(os.path.join(reports_dir, "pruning_test_report.csv"), index=False)

    print()
    print(f"Test Report   : {os.path.join(reports_dir, 'pruning_test_report.csv')}")

    return report_df, test_report_df
