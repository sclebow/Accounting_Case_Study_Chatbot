# This is a Streamlit application for an accounting case study chatbot.
# The application uses the Cloudflare Workers Service to serve chat requests and interact with the OpenAI API.

import hashlib

import requests
import streamlit as st
from openai import APIStatusError, OpenAI, RateLimitError

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

CHAT_CONTAINER_HEIGHT = 500

MODEL_OPTIONS = {
    "Gemma 4 26B A4B IT (Google)": "@cf/google/gemma-4-26b-a4b-it",
    "Qwen 3.8 27B (Alibaba)": "@cf/qwen/qwen3.8-27b",
    "Deepseek V4 Pro 0813 (Deepseek)": "@cf/deepseek-ai/deepseek-v4-pro-0813",
    "Deepseek V4 Flash 0731 (Deepseek)": "@cf/deepseek-ai/deepseek-v4-flash-0731",
    "Mistral 7B Instruct v0.2 Lora (Mistral)": "@cf/mistral/mistral-7b-instruct-v0.2-lora",
    "Llama 3.2 3B Instruct (Meta)": "@cf/llama/llama-3.2-3b-instruct",
}


def convert_pdf_to_markdown(file_name, file_bytes):
    """Convert an uploaded PDF to Markdown with Cloudflare Markdown Conversion."""
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/tomarkdown"
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {cloudflare_api_key}"},
        files={"files": (file_name, file_bytes, "application/pdf")},
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()
    results = payload.get("result", [])
    if not payload.get("success") or not results:
        raise ValueError(payload.get("errors") or "Cloudflare could not convert this PDF.")

    result = results[0]
    if result.get("format") == "error":
        raise ValueError(result.get("error") or "Cloudflare could not convert this PDF.")

    markdown = result.get("data", "").strip()
    if not markdown:
        raise ValueError("Cloudflare returned an empty document.")

    return markdown

# Default messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please upload the case study PDF and then ask your questions."}]

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

    uploaded_pdf = st.file_uploader("Upload case study PDF", type=["pdf"])
    case_study_text = ""

    if uploaded_pdf is None:
        st.session_state.pop("pdf_hash", None)
        st.session_state.pop("case_study_markdown", None)
    else:
        pdf_bytes = uploaded_pdf.getvalue()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if st.session_state.get("pdf_hash") != pdf_hash:
            try:
                with st.spinner("Converting PDF to Markdown..."):
                    case_study_text = convert_pdf_to_markdown(uploaded_pdf.name, pdf_bytes)
            except (requests.RequestException, ValueError) as error:
                st.session_state.pop("case_study_markdown", None)
                st.session_state["pdf_hash"] = pdf_hash
                error_text = str(error).lower()
                limit_reached = any(
                    phrase in error_text
                    for phrase in ("rate limit", "rate_limit", "quota", "credit", "credits")
                ) or getattr(getattr(error, "response", None), "status_code", None) in (402, 429)

                if limit_reached:
                    error_message = "The PDF conversion service has reached its usage limit. Please try again tomorrow."
                else:
                    error_message = f"Could not convert the PDF: {error}"

                st.error(error_message)
                st.toast(error_message, icon=":material/error:")
            else:
                st.session_state["pdf_hash"] = pdf_hash
                st.session_state["case_study_markdown"] = case_study_text

        case_study_text = st.session_state.get("case_study_markdown", "")
        if case_study_text:
            st.caption(f"Converted: {uploaded_pdf.name}")

with cols[1]:
    # This is the main area for interacting with the chatbot
    st.write("Chatbot interaction area")
    messages_container = st.container(border=True, height=CHAT_CONTAINER_HEIGHT)
    with messages_container:
        for message in st.session_state.get("messages"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_input := st.chat_input("Type your message here...", disabled=not case_study_text):
        print("User input received:", user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with messages_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Use the Cloudflare Workers Service to send the user input and get the response from the OpenAI API

        prompt = f"{system_prompt}\n\nCase Study Text:\n{case_study_text}\n\nUser Question:\n{user_input}"

        with messages_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("_Waiting for response..._")

                print("Sending request to Cloudflare Workers Service with prompt:", prompt)

                try:
                    response = client.chat.completions.create(
                        model=MODEL_OPTIONS[st.session_state["selected_model"]],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Case Study Text:\n{case_study_text}\n\nUser Question:\n{user_input}"}
                        ],
                        stream=True,
                    )

                    assistant_response = response_placeholder.write_stream(response)
                except (RateLimitError, APIStatusError) as error:
                    error_text = str(error).lower()
                    limit_reached = any(
                        phrase in error_text
                        for phrase in ("rate limit", "rate_limit", "quota", "credit", "credits")
                    )

                    if limit_reached or getattr(error, "status_code", None) in (402, 429):
                        error_message = "The service has reached its usage limit. Please try again tomorrow."
                    else:
                        error_message = "The service could not process your request. Please try again later."

                    response_placeholder.error(error_message)
                    st.toast(error_message, icon=":material/error:")
                else:
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_response})