from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()
class AgentState(TypedDict):
    user_request: str
    proposed_action: str
    approval_decision: str
    tool_result: str

def agent_node(state: AgentState):
    print("Agent is thinking...")

    return {
        "proposed_action": "send_email"
    }



def human_approval_node(state: AgentState):
    decision = interrupt({
        "question": "Do you approve this action?",
        "proposed_action": state["proposed_action"],
        "allowed_responses": ["approve", "reject"]
    })

    return {
        "approval_decision": decision
    }


def tool_node(state: AgentState):
    print("Tool is running...")

    if state["proposed_action"] == "send_email":
        return {
            "tool_result": "Email sent successfully."
        }

    return {
        "tool_result": "Unknown action."
    }

def reject_node(state: AgentState):
    print("Action was rejected by human.")

    return {
        "tool_result": "Action rejected. Tool was not executed."
    }

def route_after_approval(state: AgentState) -> Literal["tool_node", "reject_node"]:
    if state["approval_decision"] == "approve":
        return "tool_node"

    return "reject_node"

def ask_human_for_approval():
    while True:
        decision = input("Type approve or reject: ").strip().lower()

        if decision in ["approve", "reject"]:
            return decision

        print("Invalid input. Please type exactly: approve or reject")


if __name__ == "__main__":
    builder = StateGraph(AgentState)

    builder.add_node("agent_node", agent_node)
    builder.add_node("human_approval_node", human_approval_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("reject_node", reject_node)

    builder.add_edge(START, "agent_node")
    builder.add_edge("agent_node", "human_approval_node")

    builder.add_conditional_edges(
        "human_approval_node",
        route_after_approval
    )

    builder.add_edge("tool_node", END)
    builder.add_edge("reject_node", END)

    checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "approval-demo-1"
        }
    }

    result = graph.invoke(
        {
            "user_request": "Please send an email to the customer.",
            "proposed_action": "",
            "approval_decision": "",
            "tool_result": ""
        },
        config=config
    )

    print(result)

    print("\n--- Waiting for your approval ---")

    human_decision = ask_human_for_approval()

    result = graph.invoke(
        Command(resume=human_decision),
        config=config
    )

    print(result)