from googledriver import download_folder
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--google_drive_file_path",
                        default="https://drive.google.com/drive/folders/1tf_1I0FSbE4-H66a7bG5ueme-kVzunlr?usp=sharing")
    parser.add_argument("--download_file_location", default="../data/")
    args = parser.parse_args()

    print("Downloading requested file to folder...")
    download_folder(args.google_drive_file_path, args.download_file_location)