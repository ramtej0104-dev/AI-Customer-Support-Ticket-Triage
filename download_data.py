from datasets import load_dataset

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = ds["train"].to_pandas()
df.to_csv("support_tickets_raw.csv", index=False)