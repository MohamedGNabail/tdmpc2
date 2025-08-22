import pandas as pd
import wandb

# List of files and labels
files = {
    "JRD": "JRD.txt",
    "None": "NONE.txt",
    "STD": "STD.txt",
    "BC": "BC.txt"
}

for label, filename in files.items():
    # Start a new run for each file
    wandb.init(project="epistemic-uncertainty", name=label, reinit=True, group="epistemic")
    
    df = pd.read_csv(filename, header=None, names=["step", "value"])
    
    # Log step/value pairs with explicit step argument
    for _, row in df.iterrows():
        wandb.log({"uncertainty": row["value"]}, step=int(row["step"]))
    
    wandb.finish()
