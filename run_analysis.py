import os
from analyze_dataset import analyze_dataset
from pre_processing import preprocess_dataset
from mobileNet_data_modeling import mobileNet_data_modeling
from efficientNetLite_data_modeling import efficientnet_lite_data_modeling
from ensemble_predict import ensemble_evaluate
from knowledge_distillation import knowledge_distillation
from pruning import pruning
from quantization import quantization

reports_dir   = r"C:\Private\Private proj\DiabeticRetinopathy\notebooks\reports"
csv_path      = os.path.join(reports_dir, "dataset_analysis.csv")
os.makedirs(reports_dir, exist_ok=True)
raw_dir = r"C:\Private\Private proj\DiabeticRetinopathy\raw 200 from each sev"
processed_dir  = r"C:\Private\Private proj\DiabeticRetinopathy\processed"

# analyze_dataset (    
#     output_csv=csv_path,
#     raw_dir=raw_dir,
#     target_size=(600, 600),
#     blur_threshold=5,
#     contrast_threshold=40.0,
#     illumination_threshold=15.0,
# )

# preprocess_dataset(
#     input_csv = csv_path,
#     raw_dir=raw_dir,
#     processed_dir=processed_dir,
#     target_size=(600, 600),
#     blur_threshold=5,
# )

mobileNet_data_modeling(
    processed_dir=processed_dir,
    num_classes=5,
    epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    img_size=224,
)

efficientnet_lite_data_modeling(
    processed_dir=processed_dir,
    num_classes=5,
    epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    img_size=224,
)

ensemble_evaluate(
    processed_dir=processed_dir,
    num_classes=5,
    batch_size=32,
    img_size=224,
    mobilenet_weight=0.5,
    efficientnet_weight=0.7,
)

# knowledge_distillation(
#     processed_dir=processed_dir,
#     num_classes=5,
#     epochs=20,
#     batch_size=32,
#     learning_rate=1e-4,
#     img_size=224,
#     temperature=4.0,
#     alpha=0.7,
#     mobilenet_weight=0.5,
#     efficientnet_weight=0.7,
# )

# pruning(
#     processed_dir=processed_dir,
#     num_classes=5,
#     epochs=15,
#     batch_size=32,
#     learning_rate=1e-4,
#     img_size=224,
#     pruning_amount=0.2,
# )

# quantization(
#     processed_dir=processed_dir,
#     num_classes=5,
#     batch_size=32,
#     img_size=224,
# )