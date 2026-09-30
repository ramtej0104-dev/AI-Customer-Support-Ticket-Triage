import pandas as pd

df = pd.read_csv("support_tickets_raw.csv")

category_map = {
    "REFUND": "billing",
    "INVOICE": "billing",
    "PAYMENT": "billing",
    "SUBSCRIPTION": "billing",
    "ACCOUNT": "account",
    "ORDER": "product",
    "DELIVERY": "product",
    "SHIPPING": "product",
    "CANCEL": "product",
    "CONTACT": "service",
    "FEEDBACK": "service",
}

df["label"] = df["category"].map(category_map)

assert df["label"].isnull().sum() == 0, "Some categories didn't get mapped!"

clean_df = df[["instruction", "label"]].rename(columns={"instruction": "text"})

print("New label distribution:")
print(clean_df["label"].value_counts())

clean_df.to_csv("support_tickets_clean.csv", index=False)
print("\nSaved to support_tickets_clean.csv")