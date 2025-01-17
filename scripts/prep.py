# Prep a scope for deployment
# Currently, the main thing we will want to do is create a static file for the hover preview to fetch from
# we will make a binary file with a fixed byte range for each row in the scope

import argparse
import pandas as pd
import json

def prep_scope(directory, dataset, scope_id):
    # use json to read the metadata file
    with open(f"{directory}/{dataset}/scopes/{scope_id}.json", "r") as f:
        metadata = json.load(f)
    # use pandas to read the input file
    df = pd.read_parquet(f"{directory}/{dataset}/input.parquet")

    # extract the text column
    text_column = metadata["embedding"]["text_column"]
    text = df[text_column].tolist()
    # write the text to a binary file
    with open(f"{directory}/{dataset}/scopes/{scope_id}.bin", "wb") as f:
        for text in text:
            # Encode text and either pad or truncate to 1000 bytes
            if text is None:
                text = ""
            encoded = text.encode("utf-8")
            if len(encoded) > 1000:
                encoded = encoded[:1000]
            else:
                encoded = encoded + b' ' * (1000 - len(encoded))
            f.write(encoded)
    print(f"🔥 prepped text for scope {scope_id} for {dataset}")

def main():
    parser = argparse.ArgumentParser(description="Convert a scope to a LanceDB database")
    parser.add_argument("--directory", help="Directory containing the scope", type=str, required=True)
    parser.add_argument("--dataset", help="Name of the dataset", type=str, required=True)
    parser.add_argument("--scope_id", help="ID of the scope to convert", type=str, required=True)
    parser.add_argument("--username", help="Username of the dataset", type=str) # not used, but allows for same command structure as upload
    args = parser.parse_args()
    prep_scope(args.directory, args.dataset, args.scope_id)


if __name__ == "__main__":
    main()
