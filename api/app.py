import os
# import time
# import datetime
import modal
from modal import method, enter, Image, Secret, Volume, web_endpoint
# from logger import ParquetRequestLogger

# MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "main"
MODEL_DIR = "/models"
BIG_LIMIT = 10000000

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
    cpu=4,
    allow_concurrent_inputs=100,
    keep_warm=1, # keep one instance always?
    concurrency_limit=10,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    image=st_image,
    volumes={"/lancedb": volume, "/hf_cache": hf_model_cache},
    secrets=[
        # Secret.from_name("modal-openai-secret")
        Secret.from_name("huggingface-secret"),
    ]
)
class App:
    @enter()
    def start_engine(self):
        self.device = torch.device("cpu")
        # Initialize model cache
        self.models = {}
        # Initialize db connection pool
        self.db_connections = {}
        
    def get_model(self, model_name):
        if model_name not in self.models:
            self.models[model_name] = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        return self.models[model_name]
    
    def get_db_connection(self, db_name):
        if db_name not in self.db_connections:
            self.db_connections[db_name] = lancedb.connect(f"/lancedb/{db_name}")
        return self.db_connections[db_name]

    @web_endpoint(method="GET")
    def nn(self, query: str, model: str, db: str, scope: str):
        model = self.get_model(model)
        db = self.get_db_connection(db)
        table = db.open_table(scope)
        print(db, scope, "🔍 query", query)
        embeddings = model.encode(query, normalize_embeddings=True)
        results = table.search(embeddings).metric("cosine").select(["index"]).limit(BIG_LIMIT).to_list()
        return results
    
    @web_endpoint(method="GET")
    def scope_meta(self, db: str, scope: str):
        with open(f"/lancedb/{db}/{scope}.json", "r") as f:
            return json.load(f)

    # serve static assets from cloud storage
    # @web_endpoint(method="GET")
    # def scope_preview(self, db: str, scope: str):
    #     from fastapi.responses import Response  # Add this import at the top

    #     file_path = f"/lancedb/{db}/{scope}.png"
    #     print(f"Attempting to open file: {file_path}")
        
    #     # Check if file exists
    #     if not os.path.exists(file_path):
    #         print(f"File not found: {file_path}")
    #         return {"error": "File not found"}
    #     with open(f"/lancedb/{db}/{scope}.png", "rb") as f:
    #         return Response(
    #             content=f.read(),
    #             media_type="image/png"
    #         )
    
    @web_endpoint(method="GET")
    def scope_data(self, db: str, scope: str):
        db = lancedb.connect(f"/lancedb/{db}")
        table = db.open_table(scope)
        print(db, scope, "🥧 scope_data")
        columns = ["index","x","y","tile_index_64","cluster","raw_cluster","deleted"] #"label",
        # we just want to return all the scope rows by default
        return table.search().select(columns).limit(BIG_LIMIT).to_list()

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
        where = f"array_has(sae_indices, {feature}) AND array_element(sae_acts, cast(array_position(sae_indices, {feature}) as int)) > {threshold}"
        indices = table.search().where(where).select(["index"]).limit(BIG_LIMIT).to_list()
        return [d["index"] for d in indices]

    @web_endpoint(method="GET")
    def column_filter(self, db: str, scope: str, query: str):
        db = lancedb.connect(f"/lancedb/{db}")
        table = db.open_table(scope)
        print(db, scope, "🔍 column_filter", query)
        filters = json.loads(query)
        where_clauses = []
        for f in filters:
            if f["type"] == "eq":
                where_clauses.append(f"{f['column']} = '{f['value']}'")
            elif f["type"] == "gt":
                where_clauses.append(f"{f['column']} > {f['value']}")
            elif f["type"] == "lt":
                where_clauses.append(f"{f['column']} < {f['value']}")
            elif f["type"] == "gte":
                where_clauses.append(f"{f['column']} >= {f['value']}")
            elif f["type"] == "lte":
                where_clauses.append(f"{f['column']} <= {f['value']}")
            elif f["type"] == "in":
                values = [f"'{v}'" if isinstance(v, str) else str(v) for v in f['value']]
                where_clauses.append(f"{f['column']} IN ({','.join(values)})")
            elif f["type"] == "contains":
                where_clauses.append(f"{f['column']} LIKE '%{f['value']}%'")
        
        where_clause = " AND ".join(where_clauses)
        print("where_clause", where_clause)
        return table.search().where(where_clause).select(["index"]).limit(BIG_LIMIT).to_list()

    # @web_endpoint(method="GET")
    # def sae(self, db: str, scope: str, feature: int, threshold: float):
    #     db = lancedb.connect(f"/lancedb/{db}")
    #     table = db.open_table(scope)
    #     return table.search().where(f"{feature} in sae_features AND threshold > {threshold}").to_list()
