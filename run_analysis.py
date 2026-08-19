from analyze_dataset import analyze_dataset
# from pre_processing import preprocess_dataset
# from data_modeling import data_modeling

csv_path = "dataset_analysis.csv"
raw_dir = r"C:\Private\Private proj\DiabeticRetinopathy\raw - Copy"
processed_dir  = r"C:\Private\Private proj\DiabeticRetinopathy\processed"

analyze_dataset (    
    output_csv=csv_path,
    raw_dir=raw_dir,
    target_size=(600, 600),
    blur_threshold=5,
    contrast_threshold=40.0,
    illumination_threshold=15.0,
)

# preprocess_dataset(
#     input_csv = csv_path,
#     raw_dir=raw_dir,
#     processed_dir=processed_dir,
#     target_size=(600, 600),
#     blur_threshold=5,
# )

# data_modeling(
#     processed_dir=processed_dir,
# )