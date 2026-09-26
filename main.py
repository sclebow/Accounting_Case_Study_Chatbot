# This is a Streamlit application for an accounting case study chatbot.
# The application uses the Cloudflare Workers Service to serve chat requests and interact with the OpenAI API.

import streamlit as st
from openai import OpenAI

print("\n" * 5)
print("Starting Accounting Case Study Chatbot...")

# Set the default prompt from Streamlit secrets
default_prompt = st.secrets["DEFAULT_PROMPT"]

# Set the Cloudflare API key from Streamlit secrets
cloudflare_api_key = st.secrets["CLOUDFLARE_API_KEY"]
cloudflare_account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]

client = OpenAI(
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/v1",
    api_key=cloudflare_api_key
)

MODEL_OPTIONS = {
    "Gemma 4 26B A4B IT (Google)": "@cf/google/gemma-4-26b-a4b-it",
    "Qwen 3.8 27B (Alibaba)": "@cf/qwen/qwen3.8-27b",
    "Deepseek V4 Pro 0813 (Deepseek)": "@cf/deepseek-ai/deepseek-v4-pro-0813",
    "Deepseek V4 Flash 0731 (Deepseek)": "@cf/deepseek-ai/deepseek-v4-flash-0731",
    "Mistral 7B Instruct v0.2 Lora (Mistral)": "@cf/mistral/mistral-7b-instruct-v0.2-lora",
    "Llama 3.2 3B Instruct (Meta)": "@cf/llama/llama-3.2-3b-instruct",
}

# Default messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please paste the case study text and then ask your questions."}]

st.set_page_config(
    page_title="Accounting Case Study Chatbot",
    page_icon=":teacher:",
    layout="wide",
)

st.title("Accounting Case Study Chatbot")

with st.expander("Edit System Prompt"):
    system_prompt = st.text_area("System Prompt", value=default_prompt)

cols = st.columns([1, 4])

with cols[0]:
    selected_model = st.selectbox("Select Model", options=list(MODEL_OPTIONS.keys()))
    st.session_state["selected_model"] = selected_model

    # This is the sidebar for pasting the case study text
    case_study_text = st.text_area("Paste Case Study Text Here", height="stretch")

with cols[1]:
    # This is the main area for interacting with the chatbot
    st.write("Chatbot interaction area")
    messages_container = st.container(border=True, height=400)
    with messages_container:
        for message in st.session_state.get("messages"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_input := st.chat_input("Type your message here..."):
        print("User input received:", user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with messages_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Use the Cloudflare Workers Service to send the user input and get the response from the OpenAI API

        prompt = f"{system_prompt}\n\nCase Study Text:\n{case_study_text}\n\nUser Question:\n{user_input}"

        print("Generated prompt for OpenAI API:", prompt)

        with messages_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("_Waiting for response..._")

                print("Sending request to Cloudflare Workers Service with prompt:", prompt)

                response = client.chat.completions.create(
                    model=MODEL_OPTIONS[st.session_state["selected_model"]],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Case Study Text:\n{case_study_text}\n\nUser Question:\n{user_input}"}
                    ],
                    stream=True,
                )

                assistant_response = response_placeholder.write_stream(response)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})