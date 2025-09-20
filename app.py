import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Creating a Streamlit App
st.title("🔍Langchain - Chat with Search")
"""
In this app, we are using the 'StreamlitCallbackHandler' to display the thoughts and actions.
"""

# Creating a sidebar for settings:
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq Api key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who searches the web. How can I help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", streaming=True)
    tools = [search, wiki, arxiv]

    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
