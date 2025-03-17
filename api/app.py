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
        "transformers",
        "huggingface_hub==0.23.2",
        "latentsae==0.1.3",
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
    .env({"SAE_DISABLE_TRITON": "1"})
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
    cpu=1,
    allow_concurrent_inputs=100,
    min_containers=1, # keep one instance always
    max_containers=10,
    timeout=60 * 10,
    scaledown_window=60 * 10,
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
        # Initialize caches
        self.models = {}
        self.db_connections = {}
        self.table_connections = {}  # New cache for tables
        self.sae_models = {}
        self.transformer_models = {}

    def get_model(self, model_name):
        if model_name not in self.models:
            self.models[model_name] = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        return self.models[model_name]
    
    def get_db_connection(self, db_name):
        if db_name not in self.db_connections:
            self.db_connections[db_name] = lancedb.connect(f"/lancedb/{db_name}")
        return self.db_connections[db_name]

    def get_table(self, db_name, scope):
        cache_key = f"{db_name}/{scope}"
        if cache_key not in self.table_connections:
            db = self.get_db_connection(db_name)
            self.table_connections[cache_key] = db.open_table(scope)
        return self.table_connections[cache_key]

    def get_sae(self, model_name, k_expansion):
        model_id = f"{model_name}-{k_expansion}"
        if model_id not in self.sae_models:
            from latentsae.sae import Sae
            self.sae_models[model_id] = Sae.load_from_hub(model_name, k_expansion) 
        return self.sae_models[model_id]
    
    def get_transformer(self, model_name):
        if model_name not in self.transformer_models:
            from transformers import AutoTokenizer, AutoModel
            self.transformer_models[model_name] = {
                "tokenizer": AutoTokenizer.from_pretrained(model_name),
                "model": AutoModel.from_pretrained(model_name, trust_remote_code=True)
            }
        return self.transformer_models[model_name]

    @web_endpoint(method="GET")
    def nn(self, query: str, model: str, db: str, scope: str, results: bool = False):
        model = self.get_model(model)
        table = self.get_table(db, scope)
        print(db, scope, "🔍 query", query)
        embeddings = model.encode(query, normalize_embeddings=True)
        limit = 100 if results else BIG_LIMIT
        res = table.search(embeddings).metric("cosine").limit(limit)
        if results:
            return res.to_list()
        else:
            return res.select(["index"]).to_list()

    @web_endpoint(method="GET")
    def rows_by_index(self, db: str, scope: str, indices: str):
        table = self.get_table(db, scope)
        print(db, scope, "🚣 rows_by_index", indices)
        indices = [int(i) for i in indices.split(",")]
        # Get all column names from schema and exclude the vector column
        columns = [field.name for field in table.schema if field.name != "vector"]
        return table.to_lance().take(indices=indices, columns=columns).to_pylist()
        # return table.search().where(f"index in {indices}").select(columns).to_list()
    
    @web_endpoint(method="GET")
    def feature(self, db: str, scope: str, feature: str, threshold: float):
        table = self.get_table(db, scope)
        print(db, scope, "📌 feature", feature, threshold)
        where = f"array_has(sae_indices, {feature}) AND array_element(sae_acts, cast(array_position(sae_indices, {feature}) as int)) > {threshold}"
        # columns= {
            # "index":"index", 
            # "activation":f"array_element(sae_acts, cast(array_position(sae_indices, {feature}) as int))"
        # }
        columns = ["index", "sae_indices", "sae_acts"]
        indices = table.search().where(where).select(columns).limit(BIG_LIMIT).to_list()
        # Sort indices by activation value in descending order
        # TODO: this would probably be more perfomant inside lance (but only if there are a ton of rows for the feature)
        sorted_indices = sorted(indices, key=lambda x: x["sae_acts"][x["sae_indices"].index(int(feature))], reverse=True)
        return [d["index"] for d in sorted_indices]

    @web_endpoint(method="GET")
    def column_filter(self, db: str, scope: str, query: str):
        table = self.get_table(db, scope)
        print(db, scope, "🔍 column_filter", query)
        filters = json.loads(query)
        where_clauses = []
        for f in filters:
            # Add quotes around column names
            column_name = f'`{f["column"]}`'
            if f["type"] == "eq":
                where_clauses.append(f"{column_name} = '{f['value']}'")
            elif f["type"] == "gt":
                where_clauses.append(f"{column_name} > {f['value']}")
            elif f["type"] == "lt":
                where_clauses.append(f"{column_name} < {f['value']}")
            elif f["type"] == "gte":
                where_clauses.append(f"{column_name} >= {f['value']}")
            elif f["type"] == "lte":
                where_clauses.append(f"{column_name} <= {f['value']}")
            elif f["type"] == "in":
                values = [f"'{v}'" if isinstance(v, str) else str(v) for v in f['value']]
                where_clauses.append(f"{column_name} IN ({','.join(values)})")
            elif f["type"] == "contains":
                where_clauses.append(f"{column_name} LIKE '%{f['value']}%'")
        
        where_clause = " AND ".join(where_clauses)
        print("where_clause", where_clause)
        return table.search().where(where_clause).select(["index"]).limit(BIG_LIMIT).to_list()

    @web_endpoint(method="GET")
    def calc_embedding(self, query: str):
        import torch

        model_name = "nomic-ai/nomic-embed-text-v1.5"
        model = self.get_transformer(model_name)
        tokenizer = model["tokenizer"]
        model = model["model"]

        def mean_pooling(token_embeddings, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        text = [query]

        # Tokenize the text and move to GPU if available
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # Get token spans to preserve boundaries
        encoding = tokenizer(text, return_offsets_mapping=True)
        token_spans = encoding.offset_mapping

        print("tokens", tokens)
        print("token_spans", token_spans)

        print("tokens", tokens)
        print("DECODED",tokenizer.decode(inputs["input_ids"][0]))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get the last hidden state
        embedding = mean_pooling(outputs[0], inputs["attention_mask"])
        normalized_embeddings = torch.nn.functional.normalize(embedding, p=2, dim=1)
        normalized_hidden_states = torch.nn.functional.normalize(outputs[0], p=2, dim=1)
        return { 
            "text": text,
            "token_ids": inputs["input_ids"].cpu().numpy().tolist()[0],
            "tokens": tokens,
            "token_spans": token_spans,
            # "hidden_states": normalized_hidden_states.cpu().numpy().tolist(), 
            "hidden_states": outputs[0].cpu().numpy().tolist(),
            "attention_mask": inputs["attention_mask"].cpu().numpy().tolist()[0],
            "embedding": normalized_embeddings.cpu().numpy().tolist()[0],  # we only embed one query at a time
            "embedding_raw": embedding.cpu().numpy().tolist()[0]
        }

    @web_endpoint(method="POST")
    def calc_sae(self, item: dict):
        print("calc_sae", item.keys())

        # hack because we ran out of endpoints
        if "neighbors" in item:
            table = self.get_table(item["db"], item["scope"])
            results = table.search(item["embedding"]).metric("cosine").limit(100).to_list()
            return results

        model_name = "enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT"
        k_expansion = "64_32"
        sae = self.get_sae(model_name, k_expansion)
        if "embedding" in item:
            embedding = np.array(item["embedding"])
            embedding_tensor = torch.from_numpy(embedding).float().to(self.device)
            features = sae.encode(embedding_tensor)
            return {
                "top_indices": features.top_indices.tolist(),
                "top_acts": features.top_acts.tolist()
            }
        if "features" in item:
            top_acts = np.array(item["features"]["top_acts"])
            top_indices = np.array(item["features"]["top_indices"])
            top_acts_tensor = torch.from_numpy(top_acts).float().to(self.device)
            top_indices_tensor = torch.from_numpy(top_indices).long().to(self.device)
            embedding = sae.decode(top_acts=top_acts_tensor, top_indices=top_indices_tensor)
            return embedding.tolist()