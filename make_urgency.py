import pandas as pd

df = pd.read_csv("support_tickets_raw.csv")

urgency_map = {
    "payment_issue": "high",
    "complaint": "high",
    "recover_password": "high",
    "registration_problems": "high",
    "contact_human_agent": "high",
    "get_refund": "medium",
    "track_refund": "medium",
    "cancel_order": "medium",
    "change_order": "medium",
    "change_shipping_address": "medium",
    "track_order": "medium",
    "delivery_period": "medium",
    "check_invoice": "medium",
    "get_invoice": "medium",
    "edit_account": "medium",
    "switch_account": "medium",
    "contact_customer_service": "medium",
    "check_cancellation_fee": "medium",
    "check_payment_methods": "low",
    "check_refund_policy": "low",
    "delivery_options": "low",
    "newsletter_subscription": "low",
    "place_order": "low",
    "review": "low",
    "create_account": "low",
    "set_up_shipping_address": "low",
    "delete_account": "low",
}

df["urgency"] = df["intent"].map(urgency_map)

assert df["urgency"].isnull().sum() == 0, "Some intents were not mapped!"

urgent_words = ["urgent", "asap", "immediately", "emergency", "right now"]
has_urgent_word = df["instruction"].str.lower().str.contains("|".join(urgent_words))
df.loc[has_urgent_word, "urgency"] = "high"

clean_df = df[["instruction", "urgency"]].rename(columns={"instruction": "text"})

print("Urgency distribution:")
print(clean_df["urgency"].value_counts())

clean_df.to_csv("support_tickets_urgency.csv", index=False)
print("\nSaved to support_tickets_urgency.csv")