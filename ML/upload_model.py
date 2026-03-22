from huggingface_hub import HfApi

api = HfApi()

# api.upload_file(
#     path_or_fileobj=r"D:\Football Event Detection\ML\checkpoints\best_acc_0.6627.pth",
#     path_in_repo="best_acc_0.6627.pth",
#     repo_id="MaharshiJoshi/football-event-detection",
#     repo_type="model"
# )

# print("Model Uploaded.")


api.upload_folder(
    folder_path=r"D:\Football Event Detection\Samples",
    repo_id="MaharshiJoshi/Football-event-samples",
    repo_type="dataset"
)
