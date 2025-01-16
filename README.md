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

Then prep the scope:
This will create a static binary file of the text that can be used to cheaply fetch tooltip text
```bash
python scripts/prep.py --directory ~/latent-scope-demo --dataset ls-datavis-misunderstood --scope_id scopes-001
```

Then upload the lancedb folder to the modal volume:

```bash
modal volume put lancedb ~/latent-scope-demo/ls-datavis-misunderstood/lancedb enjalot/ls-datavis-misunderstood
```

upload the json and binary files to a cloud storage bucket
```bash
gsutil -m cp ~/latent-scope-demo/ls-datavis-misunderstood/scopes/scopes-001.json gs://fun-data/latent-scope/demos/enjalot/ls-datavis-misunderstood/scopes/scopes-001.json
gsutil -m cp ~/latent-scope-demo/ls-datavis-misunderstood/scopes/scopes-001.bin gs://fun-data/latent-scope/demos/enjalot/ls-datavis-misunderstood/scopes/scopes-001.bin
```

Then deploy the app:
# for development
modal serve api/app.py
# for production
modal deploy api/app.py
```

