from excog_trajectory.data import download_nhanes_data

try:
    file_path = download_nhanes_data()
    print(f"Successfully downloaded file to {file_path}")
except Exception as e:
    print(f"Error downloading file: {e}")