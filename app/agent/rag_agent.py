from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.rag.pipeline import rag_answer
from app.config.settings import CONFIDENCE_THRESHOLD
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


class AgentState(TypedDict):
    query: str
    rag_result: Optional[dict]
    final_answer: Optional[str]
    confidence: Optional[float]



def calculator_tool(expression: str) -> str:
    allowed_chars = "0123456789+-*/(). "

    if not all(c in allowed_chars for c in expression):
        return "Invalid mathematical expression."

    try:
        return str(eval(expression))
    except Exception:
        return "Error evaluating expression."



def web_search_tool(query: str) -> str:
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )

        
        results = response.get("results", [])
        if not results:
            return "No relevant information found online."

        summaries = []
        for r in results:
            summaries.append(f"- {r.get('content', '')}")

        return "Recent web information:\n" + "\n".join(summaries)

    except Exception as e:
        return f"Web search failed: {str(e)}"



def router_node(state: AgentState):
    return state



def entry_decision(state: AgentState):
    q = state["query"].lower()

    
    if any(char.isdigit() for char in q):
        return "calculator"

    
    web_keywords = ["latest", "today", "current", "news", "recent"]
    if any(word in q for word in web_keywords):
        return "web_search"

    
    return "retrieve"



def calculator_node(state: AgentState):
    result = calculator_tool(state["query"])
    return {
        **state,
        "final_answer": result,
        "confidence": 1.0,
    }



def web_search_node(state: AgentState):
    result = web_search_tool(state["query"])
    return {
        **state,
        "final_answer": result,
        "confidence": 0.9,
    }



def retrieve_node(state: AgentState, retriever):
    result = rag_answer(
        query=state["query"],
        retriever=retriever,
    )

    return {
        **state,
        "rag_result": result,
        "confidence": result["confidence"],
    }



def rag_decision(state: AgentState):
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        return "answer"
    return "refuse"



def answer_node(state: AgentState):
    return {
        **state,
        "final_answer": state["rag_result"]["answer"],
    }



def refuse_node(state: AgentState):
    return {
        **state,
        "final_answer": "I don’t have enough reliable information to answer this.",
    }



def build_agent(retriever):
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("answer", answer_node)
    graph.add_node("refuse", refuse_node)

    # Entry
    graph.set_entry_point("router")

    # Entry routing
    graph.add_conditional_edges(
        "router",
        entry_decision,
        {
            "calculator": "calculator",
            "web_search": "web_search",
            "retrieve": "retrieve",
        },
    )

    # RAG routing
    graph.add_conditional_edges(
        "retrieve",
        rag_decision,
        {
            "answer": "answer",
            "refuse": "refuse",
        },
    )

    # Ends
    graph.add_edge("calculator", END)
    graph.add_edge("web_search", END)
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)

    return graph.compile()
