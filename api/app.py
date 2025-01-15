import os
# import time
# import datetime
import modal
from modal import method, enter, Image, Secret, Volume, web_endpoint
# from logger import ParquetRequestLogger

# MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "main"
MODEL_DIR = "/models"

volume = Volume.from_name("lancedb", create_if_missing=False)
hf_model_cache = Volume.from_name("hf_model_cache", create_if_missing=True)

# def download_model_to_image(model_dir, model_name, model_revision):
#     from huggingface_hub import snapshot_download
#     from transformers.utils import move_cache

#     os.makedirs(model_dir, exist_ok=True)

#     snapshot_download(
#         repo_id=model_name,
#         revision=model_revision,
#         local_dir=model_dir,
#         ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
#     )
#     move_cache()

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "numpy",
        "pyarrow",
        "sentence-transformers==3.2.0",
        "huggingface_hub==0.23.2",
        "einops==0.7.0",
        "hf-transfer==0.1.6",
        "lancedb",
        "openai",
        "outlines==0.1.1",
        "fastapi",
        "uuid"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"HF_HOME": "/hf_model_cache"})
    # .run_function(
    #     download_model_to_image,
    #     timeout=60 * 20,
    #     kwargs={
    #         "model_dir": MODEL_DIR,
    #         "model_name": MODEL_ID,
    #         "model_revision": MODEL_REVISION,
    #     },
    #     secrets=[
    #         Secret.from_name("huggingface-secret"),
    #     ]
    # )
)

with st_image.imports():
    import os
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    import lancedb
    import json

app = modal.App("latent-scope-api")

@app.cls(
    cpu=2,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=1,
    image=st_image,
    volumes={"/lancedb": volume, "/hf_cache": hf_model_cache},
    secrets=[
        # Secret.from_name("modal-openai-secret")
        Secret.from_name("huggingface-secret"),
    ]
)
class App:
    @enter()
    # async def start_engine(self):
    def start_engine(self):
        # self.device = torch.device("cuda")
        self.device = torch.device("cpu")
        # print("🥶 cold starting inference")
        # start = time.monotonic_ns()
        # duration_s = (time.monotonic_ns() - start) / 1e9
        # print(f"🏎️ engine started in {duration_s:.0f}s")
        
   
    @web_endpoint(method="GET")
    # async def query(self, query: str):
    def nn(self, query: str, model: str, db: str, scope: str):
        # query: text to embed and search with
        # model: nomic-ai/nomic-embed-text-v1.5
        # db: user/dataset
        # scope: scopes-00X
        model = SentenceTransformer(model, trust_remote_code=True, device=self.device)
        db = lancedb.connect(f"/lancedb/{db}")
        table = db.open_table(scope)
        print(db, scope, "🔍 query", query)
        embeddings = model.encode(query, normalize_embeddings=True)
        results = table.search(embeddings).metric("cosine").limit(10).to_list()
        return results
    
    @web_endpoint(method="GET")
    def scope_meta(self, db: str, scope: str):
        with open(f"/lancedb/{db}/{scope}.json", "r") as f:
            return json.load(f)

    @web_endpoint(method="GET")
    def scope_preview(self, db: str, scope: str):
        from fastapi.responses import Response  # Add this import at the top

        file_path = f"/lancedb/{db}/{scope}.png"
        print(f"Attempting to open file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {"error": "File not found"}
        with open(f"/lancedb/{db}/{scope}.png", "rb") as f:
            return Response(
                content=f.read(),
                media_type="image/png"
            )
    
    @web_endpoint(method="GET")
    def scope_data(self, db: str, scope: str):
        db = lancedb.connect(f"/lancedb/{db}")
        table = db.open_table(scope)
        print(db, scope, "🥧 scope_data")
        columns = ["index","x","y","tile_index_64","cluster","raw_cluster","label","deleted"]
        # we just want to return all the scope rows by default
        return table.search().select(columns).limit(10000000).to_list()

    @web_endpoint(method="GET")
    def rows_by_index(self, db: str, scope: str, indices: str):
        db = lancedb.connect(f"/lancedb/{db}")
        table = db.open_table(scope)
        print(db, scope, "🚣 rows_by_index", indices)
        indices = [int(i) for i in indices.split(",")]
        # Get all column names from schema and exclude the vector column
        columns = [field.name for field in table.schema if field.name != "vector"]
        return table.to_lance().take(indices=indices, columns=columns).to_pylist()
        # return table.search().where(f"index in {indices}").select(columns).to_list()
    
    @web_endpoint(method="GET")
    def feature(self, db: str, scope: str, feature: str, threshold: float):
        db = lancedb.connect(f"/lancedb/{db}")
        table = db.open_table(scope)
        print(db, scope, "📌 feature", feature, threshold)
        # table.search().where("(array_has(sae_indices, 746)) AND (array_element(sae_acts, 1) > .1)").select(columns).limit(1).to_list()
        indices = table.search().where(f"array_has(sae_indices, {feature}) AND array_element(sae_acts, cast(array_position(sae_indices, {feature}) as int)) > {threshold}").select(["index"]).to_list()
        return [d["index"] for d in indices]

    # @web_endpoint(method="GET")
    # def sae(self, db: str, scope: str, feature: int, threshold: float):
    #     db = lancedb.connect(f"/lancedb/{db}")
    #     table = db.open_table(scope)
    #     return table.search().where(f"{feature} in sae_features AND threshold > {threshold}").to_list()
