import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0
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


def distillation_loss(student_logits, teacher_logits, labels, num_classes, temperature, alpha):
    """
    KD loss = alpha  * soft_loss  (KL divergence against teacher soft targets)
            + (1-alpha) * hard_loss (ordinal loss against true labels)

    Temperature softens both distributions so the student learns from
    the teacher's confidence pattern, not just the argmax.
    T² rescales the KL term back to the same magnitude as the hard loss.
    """

    # Soft targets: temperature-scaled distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits  / temperature, dim=1)

    soft_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction="batchmean"
    ) * (temperature ** 2)

    # Hard targets: ordinal-aware loss against ground truth
    hard_loss = ordinal_severity_loss(student_logits, labels, num_classes)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def _load_teacher(processed_dir, num_classes, device):
    """Load both teacher models and return them in eval mode."""

    mobilenet_path    = os.path.join(processed_dir, "mobilenetv3_baseline.pth")
    efficientnet_path = os.path.join(processed_dir, "efficientnet_lite_baseline.pth")

    teacher_mobile = mobilenet_v3_large(weights=None)
    teacher_mobile.classifier[-1] = nn.Linear(
        teacher_mobile.classifier[-1].in_features,
        num_classes
    )
    teacher_mobile.load_state_dict(
        torch.load(mobilenet_path, map_location=device)
    )
    teacher_mobile.to(device).eval()

    teacher_eff = efficientnet_b0(weights=None)
    teacher_eff.classifier[-1] = nn.Linear(
        teacher_eff.classifier[-1].in_features,
        num_classes
    )
    teacher_eff.load_state_dict(
        torch.load(efficientnet_path, map_location=device)
    )
    teacher_eff.to(device).eval()

    return teacher_mobile, teacher_eff


def knowledge_distillation(
        processed_dir,
        num_classes=5,
        epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        img_size=224,
        temperature=4.0,
        alpha=0.7,
        mobilenet_weight=0.5,
        efficientnet_weight=0.5,
):
    """
    Trains a MobileNetV3 Small student model by distilling knowledge
    from the MobileNetV3 Large + EfficientNet-B0 teacher ensemble.

    temperature    : softens probability distributions (higher = softer)
    alpha          : weight for soft KL loss; (1-alpha) for hard ordinal loss
    mobilenet_weight / efficientnet_weight : ensemble teacher weights (should sum to 1.0)
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
    # Teacher ensemble (frozen)
    # -----------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_mobile, teacher_eff = _load_teacher(processed_dir, num_classes, device)

    # -----------------------------------------
    # Student: MobileNetV3 Small
    # -----------------------------------------

    student = mobilenet_v3_small(weights="DEFAULT")
    student.classifier[-1] = nn.Linear(
        student.classifier[-1].in_features,
        num_classes
    )
    student = student.to(device)

    # -----------------------------------------
    # Optimizer
    # -----------------------------------------

    optimizer = torch.optim.Adam(
        student.parameters(),
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
    print("KNOWLEDGE DISTILLATION: MobileNetV3 Small Student")
    print("=" * 60)
    print(f"\nDevice              : {device}")
    print(f"Train Samples       : {len(train_dataset)}")
    print(f"Val Samples         : {len(val_dataset)}")
    print(f"Classes             : {train_dataset.classes}")
    print(f"Epochs              : {epochs}")
    print(f"Batch Size          : {batch_size}")
    print(f"Learning Rate       : {learning_rate}")
    print(f"Temperature         : {temperature}")
    print(f"Alpha (soft/hard)   : {alpha} / {round(1.0 - alpha, 2)}")
    print(f"Teacher weights     : MobileNet={mobilenet_weight}  EfficientNet={efficientnet_weight}")
    print()

    log_records      = []
    best_val_loss    = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):

        student.train()

        running_loss = 0.0
        correct      = 0
        total        = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            student_logits = student(images)

            # Teacher ensemble logits (weighted average, no grad)
            with torch.no_grad():
                teacher_logits = (
                    mobilenet_weight    * teacher_mobile(images) +
                    efficientnet_weight * teacher_eff(images)
                )

            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                num_classes,
                temperature,
                alpha
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(student_logits, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total

        # -----------------------------------------
        # Validation
        # -----------------------------------------

        student.eval()

        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                labels  = labels.to(device)

                student_logits = student(images)
                teacher_logits = (
                    mobilenet_weight    * teacher_mobile(images) +
                    efficientnet_weight * teacher_eff(images)
                )

                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    num_classes,
                    temperature,
                    alpha
                )

                val_loss    += loss.item() * images.size(0)
                _, predicted = torch.max(student_logits, 1)
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

        # Save best model checkpoint when validation loss improves
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    # -----------------------------------------
    # Save student model & training report
    # -----------------------------------------

    model_path = os.path.join(processed_dir, "student_distilled.pth")

    # Restore best weights (lowest val loss) before saving and testing
    if best_model_state is not None:
        student.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    torch.save(student.state_dict(), model_path)

    report_df = pd.DataFrame(log_records)
    report_df.to_csv(os.path.join(reports_dir, "distillation_modeling_report.csv"), index=False)

    print()
    print("=" * 60)
    print("DISTILLATION COMPLETE")
    print("=" * 60)
    print(f"\nStudent Saved : {model_path}")
    print(f"Report Saved  : {os.path.join(reports_dir, 'distillation_modeling_report.csv')}")

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

    student.eval()

    test_loss    = 0.0
    test_correct = 0
    test_total   = 0

    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = student(images)
            loss    = ordinal_severity_loss(outputs, labels, num_classes)

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

    test_report_df.to_csv(os.path.join(reports_dir, "distillation_test_report.csv"), index=False)

    print()
    print(f"Test Report   : {os.path.join(reports_dir, 'distillation_test_report.csv')}")

    return report_df, test_report_df
