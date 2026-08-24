import os
import tempfile
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms


def ordinal_severity_loss(outputs, labels, num_classes=5):
    ce_loss  = torch.nn.functional.cross_entropy(outputs, labels)
    severity = torch.arange(num_classes, dtype=torch.float32, device=outputs.device)
    labels_f        = labels.float().unsqueeze(1)
    distances       = (severity - labels_f).abs()
    probs           = torch.softmax(outputs, dim=1)
    ordinal_penalty = (distances * probs).sum(dim=1).mean()
    return ce_loss + ordinal_penalty


def _saved_size_mb(model):
    """Measure actual saved state dict size on disk."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / 1024 / 1024
    os.unlink(tmp_path)
    return size_mb


def quantization(
        processed_dir,
        num_classes=5,
        batch_size=32,
        img_size=224,
        num_calibration_batches=50,
):
    """
    Applies Post-Training Dynamic Quantization to the pruned student model.

    Dynamic quantization converts Linear layer weights FP32 -> INT8 at load time.
    No calibration or retraining required.

    num_calibration_batches : reserved for future upgrade to static quantization.

    Saves:
      student_quantized.pth  — quantized state dict
      student_quantized.ptl  — TorchScript Lite for Android/iOS deployment
    """

    # -----------------------------------------
    # Dataset (val + test only — no training needed)
    # -----------------------------------------

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = ImageFolder(
        root=os.path.join(processed_dir, "val"),
        transform=eval_transform
    )
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------------------
    # Load pruned student model
    # Quantized ops are CPU-only
    # -----------------------------------------

    device = torch.device("cpu")

    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(
        torch.load(os.path.join(processed_dir, "student_pruned.pth"), map_location=device)
    )
    model.eval()

    size_fp32 = _saved_size_mb(model)

    # -----------------------------------------
    # Apply dynamic quantization (FP32 -> INT8)
    #
    # Weights of all nn.Linear layers stored as INT8.
    # Activations quantized on-the-fly at inference.
    # Works on any model without layer fusion or calibration.
    # -----------------------------------------

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    size_int8      = _saved_size_mb(quantized_model)
    size_reduction = (1.0 - size_int8 / size_fp32) * 100

    # -----------------------------------------
    # Validation evaluation (quantized)
    # -----------------------------------------

    print("=" * 60)
    print("QUANTIZATION: MobileNetV3 Small Student (Dynamic INT8)")
    print("=" * 60)
    print(f"\nDevice              : {device}")
    print(f"Val Samples         : {len(val_dataset)}")
    print(f"Classes             : {val_dataset.classes}")
    print(f"Model Size FP32     : {size_fp32:.2f} MB")
    print(f"Model Size INT8     : {size_int8:.2f} MB")
    print(f"Size Reduction      : {size_reduction:.1f}%")
    print()

    criterion = ordinal_severity_loss

    quantized_model.eval()

    val_loss    = 0.0
    val_correct = 0
    val_total   = 0
    val_class_correct = [0] * num_classes
    val_class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in val_loader:
            outputs      = quantized_model(images)
            loss         = criterion(outputs, labels, num_classes)
            val_loss    += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total   += labels.size(0)
            for label, pred in zip(labels, predicted):
                val_class_total[label.item()]   += 1
                val_class_correct[label.item()] += (pred == label).item()

    val_loss_avg = val_loss / val_total
    val_acc      = val_correct / val_total

    print("=" * 60)
    print("VAL RESULTS (quantized)")
    print("=" * 60)
    print(f"\nVal Samples   : {val_total}")
    print(f"Val Loss      : {val_loss_avg:.4f}")
    print(f"Val Accuracy  : {val_acc:.4f}")
    print()
    print("Per-Class Accuracy:")
    for i in range(num_classes):
        if val_class_total[i] > 0:
            cls_acc = val_class_correct[i] / val_class_total[i]
            print(f"  Severity {i}  : {cls_acc:.4f}  ({val_class_correct[i]}/{val_class_total[i]})")

    # -----------------------------------------
    # Save quantized model + TorchScript Lite
    # -----------------------------------------

    quantized_path = os.path.join(processed_dir, "student_quantized.pth")
    torch.save(quantized_model.state_dict(), quantized_path)

    lite_path     = os.path.join(processed_dir, "student_quantized.ptl")
    example_input = torch.randn(1, 3, img_size, img_size)
    traced        = torch.jit.trace(quantized_model, example_input)
    traced._save_for_lite_interpreter(lite_path)

    # -----------------------------------------
    # Test Evaluation
    # -----------------------------------------

    test_dataset = ImageFolder(
        root=os.path.join(processed_dir, "test"),
        transform=eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss    = 0.0
    test_correct = 0
    test_total   = 0
    test_class_correct = [0] * num_classes
    test_class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            outputs       = quantized_model(images)
            loss          = criterion(outputs, labels, num_classes)
            test_loss    += loss.item() * images.size(0)
            _, predicted  = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total   += labels.size(0)
            for label, pred in zip(labels, predicted):
                test_class_total[label.item()]   += 1
                test_class_correct[label.item()] += (pred == label).item()

    test_loss_avg = test_loss / test_total
    test_acc      = test_correct / test_total

    print()
    print("=" * 60)
    print("TEST RESULTS (quantized)")
    print("=" * 60)
    print(f"\nTest Samples  : {test_total}")
    print(f"Test Loss     : {test_loss_avg:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print()
    print("Per-Class Accuracy:")
    for i in range(num_classes):
        if test_class_total[i] > 0:
            cls_acc = test_class_correct[i] / test_class_total[i]
            print(f"  Severity {i}  : {cls_acc:.4f}  ({test_class_correct[i]}/{test_class_total[i]})")

    # -----------------------------------------
    # Save report
    # -----------------------------------------

    test_report_df = pd.DataFrame([{
        "size_fp32_mb"  : round(size_fp32, 2),
        "size_int8_mb"  : round(size_int8, 2),
        "size_reduction": round(size_reduction, 1),
        "val_loss"      : round(val_loss_avg, 4),
        "val_accuracy"  : round(val_acc, 4),
        "test_loss"     : round(test_loss_avg, 4),
        "test_accuracy" : round(test_acc, 4),
        **{
            f"severity_{i}_accuracy": round(
                test_class_correct[i] / test_class_total[i], 4
            ) if test_class_total[i] > 0 else None
            for i in range(num_classes)
        }
    }])

    test_report_df.to_csv(os.path.join(reports_dir, "quantization_test_report.csv"), index=False)

    print()
    print("=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"\nQuantized Model : {quantized_path}")
    print(f"Mobile Export   : {lite_path}")
    print(f"Report Saved    : {os.path.join(reports_dir, 'quantization_test_report.csv')}")

    return test_report_df