from datasets import load_dataset

print("Downloading PubChem 10M from Hugging Face...")
#downloads the pre-cleaned 10 million SMILES dataset
dataset = load_dataset("sagawa/pubchem-10m-canonicalized")

#convert it into a standard Pandas DataFrame
df = dataset['train'].to_pandas()

#save computer as CSV file
df.to_csv("pubchem_10m.csv", index=False)
print("Success! Saved as 'pubchem_10m.csv'")