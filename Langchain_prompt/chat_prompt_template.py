from langchain_core.prompts import ChatPromptTemplate,MessagePlaceholder, MessagesPlaceholder

chat_prompt = ChatPromptTemplate(
    [
        ("system","you are a helpful customer support agent"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{query}")
    ]
)
prompt = chat_prompt.invoke({'domain':"cricket", 'topic':"how to bowl a yorker"})
print(prompt)