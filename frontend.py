from typing import Set

from backend.core import run_llm
import streamlit
from streamlit_chat import message

streamlit.header("CB AMA bot")

prompt = streamlit.text_input("Prompt", placeholder="Enter your prompt about connector builder here..")

if "prompt_history" not in streamlit.session_state:
    streamlit.session_state["prompt_history"] = []

if "chat_history" not in streamlit.session_state:
    streamlit.session_state["chat_history"] = []


def create_source_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}, {source}\n"
    return sources_string


if prompt:
    with streamlit.spinner("Generating Response..."):
        generated_response = run_llm(query=prompt)
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
        formatted_response = f"{generated_response['result']}\n\n {create_source_string(sources)}"
        streamlit.session_state["prompt_history"].append(prompt)
        streamlit.session_state["chat_history"].append(formatted_response)

if streamlit.session_state["chat_history"]:
    for generated_response, prompt in zip(streamlit.session_state["chat_history"],
                                          streamlit.session_state["prompt_history"]):
        message(prompt, is_user=True)
        message(generated_response)