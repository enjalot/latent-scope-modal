import argparse
import os

def upload_scope(username, directory, dataset, scope_id):
    # upload the json and binary files to a cloud storage bucket
    os.system(f"modal volume rm lancedb {username}/{dataset} -r")
    os.system(f"modal volume put lancedb {directory}/{dataset}/lancedb {username}/{dataset}")
    cpcmd = f"gsutil -m cp {directory}/{dataset}/scopes/{scope_id}.json gs://fun-data/latent-scope/demos/{username}/{dataset}/{scope_id}.json"
    print(cpcmd)
    os.system(cpcmd)
    cpcmd = f"gsutil -m cp {directory}/{dataset}/scopes/{scope_id}.bin gs://fun-data/latent-scope/demos/{username}/{dataset}/{scope_id}.bin"
    print(cpcmd)
    os.system(cpcmd)
    # os.system(f"gsutil -m cp {directory}/{dataset}/scopes/{scope_id}.png gs://fun-data/latent-scope/demos/{username}/{dataset}/{scope_id}.png")

def main():
    parser = argparse.ArgumentParser(description="Upload a scope to a cloud storage bucket")
    parser.add_argument("--username", help="Username of the dataset", type=str, required=True)
    parser.add_argument("--directory", help="Directory containing the scope", type=str, required=True)
    parser.add_argument("--dataset", help="Name of the dataset", type=str, required=True)
    parser.add_argument("--scope_id", help="ID of the scope to upload", type=str, required=True)
    args = parser.parse_args()
    upload_scope(args.username, args.directory, args.dataset, args.scope_id)

if __name__ == "__main__":
    main()