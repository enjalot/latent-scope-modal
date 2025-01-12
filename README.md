# Latent Scope deployed API

Using [Modal](https://modal.com/) to deploy a read-only API for hosting scopes.

## Uploading a scope

TODO: this should be done by a user, but for now we are manually uploading

Going to assume HF usernames, and that when a user publishes a dataset to HF datasets using latent scope we will follow that convention. Any scopes they've "exported" to lancedb will be in the lancedb folder of the dataset.
In the future we will probably default to using lancedb instead of parquest in latent scope.
for now you'll need to export:
```bash
python latentscope/scripts/export_lance.py --directory ~/latent-scope-demo --dataset ls-datavis-misunderstood --scope_id scopes-001 --metric cosine 
```

Then upload the lancedb folder to the modal volume:

```bash
modal volume put lancedb ~/latent-scope-demo/ls-datavis-misunderstood/lancedb enjalot/ls-datavis-misunderstood
```

```bash
# for development
modal serve api/app.py
# for production
modal deploy api/app.py
```

