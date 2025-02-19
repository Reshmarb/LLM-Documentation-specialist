import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import BaseLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import LLMResult
from pydantic import BaseModel, Field
import os
import tempfile
import replicate
from dotenv import load_dotenv
from typing import Any, List, Optional


# Load environment variables
load_dotenv()

os.environ["REPLICATE_API_TOKEN"] = "r8_SVj0kO3GRGZSC0fH0QCT3xiI6N7NjB52igL4i"

# Custom Replicate LLM class with proper Pydantic declarations
class ReplicateLLM(BaseLLM, BaseModel):
    """Custom LLM class to use Replicate API for inference."""
    
    model_name: str = Field(..., alias="model")
    api_token: str = Field(...)
    temperature: float = Field(default=0.01)
    max_new_tokens: int = Field(default=500)
    top_p: float = Field(default=1)
    client: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = replicate.Client(api_token=self.api_token)

    def _generate(self, prompts: List[str], **kwargs) -> LLMResult:
        try:
            responses = []
            for prompt in prompts:
                output = self.client.run(
                    self.model_name,
                    input={
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_new_tokens,
                        "top_p": self.top_p,
                    }
                )
                responses.append("".join(output))
            return LLMResult(generations=[[{"text": r}] for r in responses])
        except Exception as e:
            return LLMResult(generations=[[{"text": f"Error: {str(e)}"}]])

    @property
    def _llm_type(self) -> str:
        return "replicate"

def initialize_session_state():
    """Initialize session state for chat history and generated messages."""
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your documents 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    """Process the user query and fetch response from the chain."""
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    """Display the entire chat history."""
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    """Create the conversational chain with Replicate LLM."""
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        st.error("❌ REPLICATE_API_TOKEN not found in environment variables")
        return None

    llm = ReplicateLLM(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        api_token=api_token,
        temperature=0.01,
        max_new_tokens=500
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return chain

def main():
    """Main function to run the app."""
    initialize_session_state()
    st.title("Multi-Document Specialist")
    
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        documents = []
        for file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file_path)
                elif file.name.endswith((".docx", ".doc")):
                    loader = Docx2txtLoader(temp_file_path)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(temp_file_path)

                if loader:
                    documents.extend(loader.load())
                os.remove(temp_file_path)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

        if not documents:
            st.error("No valid documents processed")
            return

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)

        chain = create_conversational_chain(vector_store)
        if chain:
            display_chat_history(chain)

if __name__ == "__main__":
    main()