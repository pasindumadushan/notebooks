import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_large, efficientnet_b0
import torchvision.transforms as transforms


def load_mobilenet(model_path, num_classes, device):
    model = mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_efficientnet(model_path, num_classes, device):
    model = efficientnet_b0(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def ensemble_evaluate(
        processed_dir,
        num_classes=5,
        batch_size=32,
        img_size=224,
        mobilenet_weight=0.5,
        efficientnet_weight=0.5,
):
    """
    Loads both saved baseline models and evaluates them as a soft-voting ensemble.

    mobilenet_weight + efficientnet_weight should sum to 1.0.
    Default is equal weighting (0.5 / 0.5).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mobilenet_path    = os.path.join(processed_dir, "mobilenetv3_baseline.pth")
    efficientnet_path = os.path.join(processed_dir, "efficientnet_lite_baseline.pth")

    print("=" * 60)
    print("ENSEMBLE EVALUATION")
    print("=" * 60)
    print(f"\nDevice              : {device}")
    print(f"MobileNet weight    : {mobilenet_weight}")
    print(f"EfficientNet weight : {efficientnet_weight}")
    print(f"\nLoading MobileNetV3 Large  from : {mobilenet_path}")
    print(f"Loading EfficientNet-B0    from : {efficientnet_path}")

    mobilenet    = load_mobilenet(mobilenet_path, num_classes, device)
    efficientnet = load_efficientnet(efficientnet_path, num_classes, device)

    # -----------------------------------------
    # Test dataset
    # -----------------------------------------

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

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

    print(f"\nTest Samples        : {len(test_dataset)}")
    print(f"Classes             : {test_dataset.classes}")
    print()

    # -----------------------------------------
    # Ensemble inference
    # -----------------------------------------

    correct       = 0
    total         = 0
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            probs_mobilenet    = F.softmax(mobilenet(images), dim=1)
            probs_efficientnet = F.softmax(efficientnet(images), dim=1)

            # Weighted average of probability distributions
            avg_probs = (
                mobilenet_weight    * probs_mobilenet +
                efficientnet_weight * probs_efficientnet
            )

            predicted = torch.argmax(avg_probs, dim=1)

            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

            for label, pred in zip(labels, predicted):
                class_total[label.item()]   += 1
                class_correct[label.item()] += (pred == label).item()

    overall_acc = correct / total

    print("=" * 60)
    print("ENSEMBLE TEST RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy : {overall_acc:.4f}  ({correct}/{total})")
    print()
    print("Per-Class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            cls_acc = class_correct[i] / class_total[i]
            print(f"  Severity {i}  : {cls_acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    # -----------------------------------------
    # Save report
    # -----------------------------------------

    report = [{
        "mobilenet_weight"    : mobilenet_weight,
        "efficientnet_weight" : efficientnet_weight,
        "overall_accuracy"    : round(overall_acc, 4),
        **{
            f"severity_{i}_accuracy": round(
                class_correct[i] / class_total[i], 4
            ) if class_total[i] > 0 else None
            for i in range(num_classes)
        }
    }]

    report_df = pd.DataFrame(report)
    report_df.to_csv("ensemble_test_report.csv", index=False)

    print()
    print(f"Report Saved     : ensemble_test_report.csv")

    return report_df


if __name__ == "__main__":

    processed_dir = r"C:\Private\Private proj\DiabeticRetinopathy\processed"

    # Equal weighting — change to e.g. (0.4, 0.6) to favour EfficientNet
    ensemble_evaluate(
        processed_dir=processed_dir,
        num_classes=5,
        batch_size=32,
        img_size=224,
        mobilenet_weight=0.5,
        efficientnet_weight=0.5,
    )
