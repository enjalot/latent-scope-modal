import os
import argparse

def oneshot(username, directory, dataset, scope_id):
    os.system(f"python scripts/prep.py --username {username} --directory {directory} --dataset {dataset} --scope_id {scope_id}")
    os.system(f"python scripts/export_lance.py --username {username} --directory {directory} --dataset {dataset} --scope_id {scope_id}")
    os.system(f"python scripts/upload.py --username {username} --directory {directory} --dataset {dataset} --scope_id {scope_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload a scope to a cloud storage bucket")
    parser.add_argument("--username", help="Username of the dataset", type=str, required=True)
    parser.add_argument("--directory", help="Directory containing the scope", type=str, required=True)
    parser.add_argument("--dataset", help="Name of the dataset", type=str, required=True)
    parser.add_argument("--scope_id", help="ID of the scope to upload", type=str, required=True)
    args = parser.parse_args()
    oneshot(args.username, args.directory, args.dataset, args.scope_id)

if __name__ == "__main__":
    main()