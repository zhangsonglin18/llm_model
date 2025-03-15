import os
import json
project_dir = os.path.dirname(os.path.abspath(__file__)).split('\src')[0]
print(project_dir)
# load eval dataset
with open(os.path.join(project_dir, "data/ft_val_dataset.json"), "r", encoding="utf-8") as f:
    eval_content = json.loads(f.read())