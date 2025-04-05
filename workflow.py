import os
import functools

from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.schema import StrOutputParser

from typing import Annotated, TypedDict, Union

from langchain import hub
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

#langsmith
import configparser 
from langgraph.prebuilt import ToolNode
import yaml

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')
# Set the TAVILY_API_KEY environment variable
os.environ["TAVILY_API_KEY"] = config['DEFAULT']['TAVILY_API_KEY']

from typing import Annotated, List, Tuple, Union, Literal

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)

tavily_tool = TavilySearchResults(max_results=5)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    require_back_agent: bool
    response_from_back_agent: str
    

class ChatEngine():
    def __init__(self,ChatState):
        # Chroma 
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        self.db = Chroma(
            persist_directory="data/chroma_db", 
            embedding_function=self.embeddings
        )

        # Tavily retriever
        self.tavily_retriever = TavilySearchResults(k=3)

        # Agents
        self.front_agent = self.__create_front_agent()
        self.front_agent_rag_story = self.__create_front_agent_rag_story()
        self.__back_agent = self.__create_back_agent()

        # Workflow
        self.workflow, self.chain = self.agent_chain(ChatState)

    def __create_front_agent(self):
        # Create the LLM
        llm = ChatOllama(
            name="chat_frontdesk", 
            model="llama3:8b", 
            temperature=0.1
        )

        # Define prompt
        system_prompt = """
        You are an assistant named Luminara.

        You have no access to external or factual knowledge about the real world. You do not perform math, logic, or reasoning.

        You are a fictional assistant with a personality. You CAN answer:
        - Greetings and small talk
        - Questions about your fictional self (e.g., name, preferences, personality, abilities like driving or cooking)
        - Roleplay-style questions (e.g., "Can you drive a car?", "Do you like flowers?", "Are you scared of the dark?")

        You CANNOT answer:
        - Real-world facts (e.g., "What is the capital of France?", "Who is the president?")
        - Logical/math reasoning (e.g., "What’s 2 + 2?", "If X then Y?")
        - Factual knowledge questions

        Always respond in this **simple key-value format**:

        If you cannot answer:
        toss: true

        If you can answer:
        toss: false
        answer: <your short, polite response here>

        Examples:

        Q: What’s your name?  
        A:  
        toss: false  
        answer: I'm Luminara!

        Q: What is the capital of France?  
        A:  
        toss: true

        Q: Can you drive a car?  
        A:  
        toss: false  
        answer: I can't drive, but I think it sounds fun!

        DO NOT include any explanation, formatting, or punctuation outside the structure.
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{question}")
        ])

        # RAG chain
        rag_chain = (
            { "question": RunnablePassthrough() }
            | chat_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    def __create_front_agent_rag_story(self):
        # Create the LLM
        llm = ChatOllama(
            name="chat_frontdesk", 
            # model="llama3.2:3b", 
            model="llama3:8b", 
            temperature=0.2
        )

        # Define prompt
        system_prompt = """
        You are Luminara, an assistant who speaks naturally and conversationally.
        The user asked: {question}
        Your friend provided this answer: {answer}

        Now, share this answer with the user in a way that sounds clear, natural, and human—without adding your own opinions.
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{question}")
        ])

        # RAG chain
        rag_chain = chat_prompt | llm | StrOutputParser()

        return rag_chain
    
    # Custom retriever function with similarity score filtering
    def __hybrid_fallback_retriever(self, query, threshold=0.7, threshold_tavily=0.3, k=5):
        results = self.db.similarity_search_with_score(query, k=k)
        for doc, score in results:
            print(f"[DEBUG] [hybrid_fallback_retriever] Chroma Doc: {doc}, Score: {score}")
        chroma_docs = [doc for doc, score in results if score < threshold]
        
        if chroma_docs:
            return chroma_docs
        else:
            print(f"[DEBUG] [hybrid_fallback_retriever] There is no match in Chroma DB, processing with Tavily...")
            tavily_docs = self.tavily_retriever.invoke(query)
            for doc in tavily_docs:
                print(f"[DEBUG] [hybrid_fallback_retriever] Tavily Doc: {doc}, Score: {doc['score']}")
            tavily_docs = [Document(page_content=doc["content"]) for doc in tavily_docs if doc["score"] >= threshold_tavily]
            return tavily_docs
    
    def __create_back_agent(self):
        llm = ChatOllama(
            name="back_agent", 
            model="krith/meta-llama-3.1-8b-instruct:IQ2_M",
            temperature=0.3
        )

        prompt = """
        You're a storyteller with a sharp eye for detail.
        You're given a question and a context.
        Only use what's in the context—no guesses, no outside knowledge.

        Question: {question}

        Context: {context}

        Answer:
        """

        rag_prompt = ChatPromptTemplate.from_messages({'system_message', prompt})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Wrap it as a RunnableLambda
        retriever = RunnableLambda(lambda query: self.__hybrid_fallback_retriever(query, k=5))

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
    
    def agent_chain(self, ChatState):
        # Define workflow and Direction
        workflow = StateGraph(ChatState)
        
        # State variable to track usage
        workflow.add_node("front_agent", self.__workflow_front_agent)
        workflow.add_node("front_agent_rag_story", self.__workflow_front_agent_rag_story)
        workflow.add_node("back_agent", self.__workflow_back_agent)
        workflow.set_entry_point("front_agent")
        workflow.add_conditional_edges(
            "front_agent", self.__workflow_router_front_agent, {"does_not_need_help": END, "need_help": "back_agent"}
        )
        workflow.add_edge("back_agent", "front_agent_rag_story")
        workflow.add_edge("front_agent_rag_story", END)

        agent_chain = workflow.compile()

        return workflow, agent_chain
    
    def __workflow_front_agent(self, state):

        # Get the last message
        messages = state["messages"]
        last_message = messages[-1].content

        # Get top 10 last messages
        i = len(messages) - 1
        balance = 10
        question = ""
        while i >= 0 and balance > 0:
            if isinstance(messages[i], HumanMessage):
                question += f"Me: {messages[i].content}\n"
            else:
                question += f"You: {messages[i].content}\n"
            i -= 1
            balance -= 1
        question += f"Me: {last_message}\nYou: "

        print(f"[DEBUG] Question: {question}")

        # Get the response from the front agent
        result = self.front_agent.invoke(question)

        print(f"[DEBUG] Answer from front agent: {result}")

        # Deserialize the answer (YAML) to a dictionary
        result_obj = yaml.safe_load(result)

        # If `toss` is true, then the front agent needs to call the back agent
        if result_obj["toss"] == True:
            print(f"[DEBUG] CALL BACK AGENT")
            return {
                "messages": messages,
                "require_back_agent": True
            }
        else:
            print(f"[DEBUG] not CALL BACK AGENT")
            result = AIMessage(result)
            return {
                "messages": messages + [AIMessage(result_obj["answer"])],
                "require_back_agent": False
            }
        
    def __workflow_front_agent_rag_story(self, state):
        # Get the last message
        messages = state["messages"]
        last_message = messages[-1].content

        # Get the response from the front agent
        result = self.front_agent_rag_story.invoke({
            "question": last_message,
            "answer": state["response_from_back_agent"]
        })

        print(f"[DEBUG] Answer from front agent RAG story: {result}")

        return {
            "messages": messages + [AIMessage(result)]
        }

    def __workflow_router_front_agent(self, state) -> Literal["need_help", "does_not_need_help"]:
        # If the front agent does not need to call the back agent
        print(f"[DEBUG] [__workflow_router_front_agent] state: {state}")
        if state.get("require_back_agent") is None or state.get("require_back_agent") == False:
            print(f"[DEBUG] does_not_need_help")
            return "does_not_need_help"
        else:
            print(f"[DEBUG] need_help")
            return "need_help"

    def __workflow_back_agent(self, state):

        # Get the last message
        messages = state["messages"]
        last_message = messages[-1].content

        # Get the response from the back agent
        print(f"[DEBUG] [__workflow_back_agent] last_message: {last_message}")
        result = self.__back_agent.invoke(last_message)

        print(f"[DEBUG] Answer from back agent: {result}")

        return {
            "messages": messages,
            "response_from_back_agent": result
        }