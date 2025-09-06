from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from src.utils import format_docs

def init_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1000, streaming: bool = True):
    llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming
    )
    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant utile pour la SGCI.
Voici l'historique des conversations :
{history}

Voici des extraits du document qui peuvent aider :
{context}

Nouvelle question : {question}
Réponds uniquement sur la base du document et de l’historique.
Si tu n’as pas la réponse dans le document, dis-le clairement.
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return llm, prompt, chain

def display_chat_history(st, collection_name: str, chat_history: dict):
    """Affiche l’historique des messages pour la collection active"""
    for user_q, bot_a in chat_history.get(collection_name, []):
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            st.write(bot_a)

def handle_user_input(st, user_input: str, active_collection: str, chat_history: dict, vector_store, chain, prompt):
    """Gère la question utilisateur, récupération du contexte et streaming de la réponse"""
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )
    docs = retriever.invoke(user_input)
    context = format_docs(docs)

    history_text = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history[active_collection]]
    )

    # afficher le message utilisateur
    with st.chat_message("user"):
        st.write(user_input)

    # afficher la réponse en streaming
    with st.chat_message("assistant"):
        def stream_response():
            messages = prompt.format_prompt(
                history=history_text,
                context=context,
                question=user_input
            ).to_messages()
            for token in chain.llm.stream(messages):
                if token.content:
                    yield token.content

        streamed_text = st.write_stream(stream_response)

    # sauvegarder dans l’historique
    chat_history[active_collection].append((user_input, streamed_text))
