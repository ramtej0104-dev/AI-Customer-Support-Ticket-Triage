import streamlit as st
from predict import triage

st.title("AI Customer Support Ticket Triage")
st.write("Paste a customer message and the model will guess which team should handle it, and how urgent it is.")

ticket = st.text_area("Customer message")

if st.button("Analyze ticket"):
    if ticket.strip() == "":
        st.warning("Please type a message first.")
    else:
        category, confidence, urgency, status = triage(ticket)

        st.subheader(f"Category: {category}")
        st.write(f"Confidence: {confidence:.0%}")

        if urgency == "high":
            st.error(f"Urgency: {urgency.upper()}")
        elif urgency == "medium":
            st.warning(f"Urgency: {urgency.upper()}")
        else:
            st.info(f"Urgency: {urgency.upper()}")

        if status == "NEEDS HUMAN REVIEW":
            st.error("Needs human review")
        else:
            st.success(f"Auto-routed to the {category} team")