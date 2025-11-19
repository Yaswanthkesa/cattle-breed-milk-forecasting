import os

def main():
    print("\n============================================")
    print(" Indian Bovine Breeds Dataset Downloader")
    print("============================================\n")

    print("This script downloads the dataset from Kaggle.")
    print("Please make sure you have:")
    print("1. A Kaggle account")
    print("2. kaggle.json API token downloaded")
    print("3. Kaggle CLI installed\n")

    # Check kaggle installation
    print("Checking Kaggle installation...\n")
    if os.system("kaggle --version") != 0:
        print("❌ Kaggle CLI not installed.")
        print("Install it using:")
        print("pip install kaggle")
        return
    
    # Create dataset folder
    os.makedirs("data/full_dataset", exist_ok=True)

    print("\nDownloading dataset... This may take time (3GB)...\n")
    os.system("kaggle datasets download -d ulnproject/indian-bovine-breeds -p data/full_dataset")

    print("\nExtracting dataset...\n")
    os.system("unzip -o data/full_dataset/indian-bovine-breeds.zip -d data/full_dataset")

    print("✔ Dataset downloaded successfully!")
    print("Location: data/full_dataset/\n")

if __name__ == "__main__":
    main()
