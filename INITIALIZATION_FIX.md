# Quick Fix: Initialize and Train BERT Model

This script will train the BERT model for 1 epoch to initialize it properly.
Once trained, the model will be saved and can be loaded by the API.

## Run this script:

```bash
cd project/backend
python train_initial.py
```

This will:
1. Load the data
2. Build the BERT model
3. Train for 1 epoch (~5-10 minutes)
4. Save the model to `bert_ner_model/`

After running this, restart the backend and the model will load successfully.
