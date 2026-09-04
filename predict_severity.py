import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms


def _build_eval_transform(img_size=224):
	return transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])


def _load_quantized_model(model_path, num_classes=5):
	"""Rebuild quantized architecture and load saved quantized state dict."""
	device = torch.device("cpu")

	model = mobilenet_v3_small(weights=None)
	model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
	model.eval()

	quantized_model = torch.quantization.quantize_dynamic(
		model,
		{nn.Linear},
		dtype=torch.qint8
	)
	quantized_model.load_state_dict(torch.load(model_path, map_location=device))
	quantized_model.eval()
	return quantized_model


def _predict_one_image(model, image_path, transform):
	image = Image.open(image_path).convert("RGB")
	image_tensor = transform(image).unsqueeze(0)

	with torch.no_grad():
		outputs = model(image_tensor)
		probs = torch.softmax(outputs, dim=1)
		predicted_idx = int(torch.argmax(probs, dim=1).item())
		confidence = float(probs[0, predicted_idx].item())

	return predicted_idx, confidence, probs.squeeze(0).tolist()


def predict_severity(
		processed_dir,
		prediction_dir,
		num_classes=5,
		img_size=224,
		max_images=4,
):
	"""
	Predicts DR severity for up to max_images from prediction_dir
	using the final quantized student model.

	Saves:
	  prediction_report.csv in notebooks/reports
	"""

	reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
	os.makedirs(reports_dir, exist_ok=True)

	quantized_model_path = os.path.join(processed_dir, "student_quantized.pth")
	if not os.path.exists(quantized_model_path):
		raise FileNotFoundError(
			f"Quantized model not found: {quantized_model_path}. Run quantization first."
		)

	if not os.path.isdir(prediction_dir):
		raise FileNotFoundError(f"Prediction folder not found: {prediction_dir}")

	valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
	image_files = [
		os.path.join(prediction_dir, name)
		for name in sorted(os.listdir(prediction_dir))
		if os.path.splitext(name.lower())[1] in valid_ext
	]

	if len(image_files) == 0:
		raise ValueError(f"No supported image files found in: {prediction_dir}")

	selected_images = image_files[:max_images]

	transform = _build_eval_transform(img_size=img_size)
	model = _load_quantized_model(quantized_model_path, num_classes=num_classes)

	print("=" * 60)
	print("SEVERITY PREDICTION (QUANTIZED STUDENT)")
	print("=" * 60)
	print(f"\nModel Path          : {quantized_model_path}")
	print(f"Prediction Folder   : {prediction_dir}")
	print(f"Total Images Found  : {len(image_files)}")
	print(f"Images Used         : {len(selected_images)}")
	print()

	per_image_rows = []
	probs_accumulator = []

	for image_path in selected_images:
		severity, confidence, probs = _predict_one_image(model, image_path, transform)
		probs_accumulator.append(probs)

		row = {
			"image_name": os.path.basename(image_path),
			"predicted_severity": severity,
			"confidence": round(confidence, 4),
		}
		for i in range(num_classes):
			row[f"prob_severity_{i}"] = round(float(probs[i]), 6)

		per_image_rows.append(row)

		print(
			f"Image: {os.path.basename(image_path):<30} "
			f"Severity: {severity}  Confidence: {confidence:.4f}"
		)

	probs_tensor = torch.tensor(probs_accumulator, dtype=torch.float32)
	mean_probs = probs_tensor.mean(dim=0)
	final_severity = int(torch.argmax(mean_probs).item())
	final_confidence = float(mean_probs[final_severity].item())
	worst_severity = int(torch.tensor([row["predicted_severity"] for row in per_image_rows]).max().item())
	disagreement = int(
		max(row["predicted_severity"] for row in per_image_rows) -
		min(row["predicted_severity"] for row in per_image_rows)
	)
	consistency_flag = "consistent" if disagreement <= 1 else "needs_review"

	print("\n" + "=" * 60)
	print("FINAL DECISION")
	print("=" * 60)
	print(f"Mean-Prob Severity  : {final_severity}")
	print(f"Mean-Prob Confidence: {final_confidence:.4f}")
	print(f"Worst-Case Severity : {worst_severity}")
	print(f"Consistency Flag    : {consistency_flag}")

	per_image_df = pd.DataFrame(per_image_rows)

	summary_row = {
		"image_name": "__summary__",
		"predicted_severity": final_severity,
		"confidence": round(final_confidence, 4),
		"worst_case_severity": worst_severity,
		"consistency_flag": consistency_flag,
		"images_used": len(selected_images),
	}
	for i in range(num_classes):
		summary_row[f"prob_severity_{i}"] = round(float(mean_probs[i].item()), 6)

	report_df = pd.concat([per_image_df, pd.DataFrame([summary_row])], ignore_index=True)

	report_path = os.path.join(reports_dir, "prediction_report.csv")
	report_df.to_csv(report_path, index=False)

	print(f"\nReport Saved        : {report_path}")

	return report_df


if __name__ == "__main__":
	processed_dir = r"C:\Private\Private proj\DiabeticRetinopathy\processed"
	prediction_dir = r"C:\Private\Private proj\DiabeticRetinopathy\notebooks\prediction images"

	predict_severity(
		processed_dir=processed_dir,
		prediction_dir=prediction_dir,
		num_classes=5,
		img_size=224,
		max_images=4,
	)
