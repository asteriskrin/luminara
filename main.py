import chainlit as cl
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable


from workflow import ChatState, ChatEngine

@cl.on_chat_start
async def on_chat_start():
    # start graph
    state = ChatState
    chat_engine = ChatEngine(ChatState)
    agent_chain = chat_engine.chain
    # save graph and state to the user session
    cl.user_session.set("graph", agent_chain)
    cl.user_session.set("state", state)

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the graph and state from the user session
    graph: Runnable = cl.user_session.get("graph")
    state = cl.user_session.get("state")

    # Append the new message to the state
    inputs = {"messages": [] + [HumanMessage(content=message.content)]}
    config = {"configurable": {"thread_id": "1"}}

    # Stream the response to the UI
    ui_message = cl.Message(content="")
    await ui_message.send()

    async for event in graph.astream(inputs, config, debug=True, stream_mode="values"):
        messages = event['messages']
        last_message = messages[-1]
        content = last_message.content
        if isinstance(last_message,HumanMessage) and last_message.content != '':
            content = last_message.content
            await ui_message.update()
        if isinstance(last_message,AIMessage) and last_message.content != '':
            content = last_message.content
            await ui_message.stream_token(token=content)
            await ui_message.update()
    
    await ui_message.update()