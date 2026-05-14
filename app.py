import streamlit as st
from chatbot_with_rag import chatbot , retriev_all_threads , create_faiss_vector_store
from langchain_core.messages import HumanMessage,AIMessage
import uuid
import tempfile
from langgraph.types import interrupt,Command
# st.session_state -> dict -> 

#utility function to generate unique thread ids
def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thraed_id=generate_thread_id()
    st.session_state['thread_id']=thraed_id
    add_chatid(st.session_state["thread_id"])
    st.session_state['message_history']=[]

def add_chatid(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get("messages", [])


# Initialize message history in session state if it doesn't exist
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"]=retriev_all_threads()
if "vector_store" not in st.session_state:
    st.session_state["vector_store"]=None
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False
add_chatid(st.session_state["thread_id"])


#sidebar
st.sidebar.title("langgraph conversational agent")
pdf_uploader = st.sidebar.file_uploader("Upload a PDF for RAG", type=["pdf"], key="pdf_uploader")
if pdf_uploader is not None:
    # Check if this is a new file upload
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != pdf_uploader.name:
        st.session_state["last_uploaded_file"] = pdf_uploader.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_uploader.read())
            temp_path = tmp_file.name
        try:
            vector_store = create_faiss_vector_store(temp_path)
            st.session_state["vector_store"] = vector_store
            st.session_state["pdf_processed"] = True
            st.sidebar.success("✅ PDF uploaded and vector store created successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing PDF: {str(e)}")
            st.session_state["pdf_processed"] = False
    else:
        if st.session_state.get("vector_store") is not None:
            st.sidebar.info("📄 PDF already loaded")
else:
    st.session_state["last_uploaded_file"] = None
    st.session_state["pdf_processed"] = False
if st.sidebar.button("new chat"):
    reset_chat()
st.sidebar.header("Conversation history")   
for thread in st.session_state["chat_threads"]: 
    if st.sidebar.button(f"Current thread id:{str(thread)}"):
        st.session_state["thread_id"]=thread
        messages=load_conversation(thread)
        temp_messages=[] 
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role="user"
            else :
                role="assistance"
            temp_messages.append({"role":role,"content":msg.content})
        st.session_state["message_history"]=temp_messages            



# loading the conversation history
if st.session_state.get("pdf_processed") and st.session_state.get("vector_store") is not None:
    st.success("🔍 Vector Store Ready - PDF Loaded!")
    with st.expander("📋 PDF Info"):
        st.write(f"**PDF File:** {st.session_state.get('last_uploaded_file', 'Unknown')}")
else:
    st.info("⚠️ No PDF loaded. Please upload a PDF in the sidebar to enable RAG feature.")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')

if user_input:
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    # first add the message to message_history
    
    with st.chat_message('assistant'):
        
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content            

    ai_message=st.write_stream(ai_only_stream)
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})    