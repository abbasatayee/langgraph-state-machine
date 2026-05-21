from dotenv import load_dotenv
from typing import TypedDict, Literal, Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# This file is often run directly while learning:
#   python "src/hitl_demo /hitl.py"
# In that mode Python does not know the parent package, so a relative import
# like `from .utils import ...` fails. Keep the package import for `python -m`,
# but fall back to importing from this file's directory for direct execution.
try:
    from .utils import export_graph_image
except ImportError:
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import export_graph_image

# load environment variables from .env file
load_dotenv()

# --------------------------------------------------
# 1. State
# --------------------------------------------------

class AgentState(TypedDict):
    user_request: str

    proposed_tool: str
    proposed_args: dict[str, Any]

    middleware_decision: str
    middleware_reason: str

    human_decision: str
    human_feedback: str

    tool_result: str


# --------------------------------------------------
# 2. Policy layer
# --------------------------------------------------

POLICY = {
    "search_customer": {
        "decision": "allow",
        "allowed_human_decisions": []
    },
    "send_email": {
        "decision": "review",
        "allowed_human_decisions": ["approve", "edit", "reject"]
    },
    "delete_customer": {
        "decision": "block",
        "allowed_human_decisions": []
    }
}


# --------------------------------------------------
# 3. Agent node
# --------------------------------------------------

def agent_node(state: AgentState):
    print("\nAGENT: Thinking about the user request...")

    # Fake agent decision.
    # Later, this can be replaced by an LLM that proposes tool calls.
    return {
        "proposed_tool": "send_email",
        "proposed_args": {
            "to": "john@example.com",
            "subject": "Refund approved",
            "body": "Your refund is approved."
        }
    }


# --------------------------------------------------
# 4. Middleware node
# --------------------------------------------------

def middleware_node(state: AgentState):
    print("\nMIDDLEWARE: Checking proposed tool call against policy...")

    tool_name = state["proposed_tool"]
    policy = POLICY.get(tool_name)

    if policy is None:
        return {
            "middleware_decision": "block",
            "middleware_reason": f"No policy found for tool: {tool_name}"
        }

    decision = policy["decision"]

    if decision == "allow":
        reason = f"{tool_name} is low-risk and allowed automatically."

    elif decision == "review":
        reason = f"{tool_name} requires human approval."

    elif decision == "block":
        reason = f"{tool_name} is blocked by policy."

    else:
        decision = "block"
        reason = f"Invalid policy decision for tool: {tool_name}"

    return {
        "middleware_decision": decision,
        "middleware_reason": reason
    }


# --------------------------------------------------
# 5. Human review node
# --------------------------------------------------

def human_review_node(state: AgentState):
    print("\nHUMAN REVIEW: Pausing graph for human decision...")

    review_payload = {
        "message": "Middleware requires human review.",
        "reason": state["middleware_reason"],
        "proposed_tool": state["proposed_tool"],
        "proposed_args": state["proposed_args"],
        "allowed_decisions": POLICY[state["proposed_tool"]]["allowed_human_decisions"]
    }

    human_response = interrupt(review_payload)

    decision = human_response.get("decision")
    feedback = human_response.get("feedback", "")
    edited_args = human_response.get("edited_args")

    updates = {
        "human_decision": decision,
        "human_feedback": feedback
    }

    if decision == "edit" and edited_args:
        updates["proposed_args"] = edited_args

    return updates


# --------------------------------------------------
# 6. Tool node
# --------------------------------------------------

def tool_node(state: AgentState):
    print("\nTOOL: Executing approved tool call...")

    tool_name = state["proposed_tool"]
    args = state["proposed_args"]

    if tool_name == "search_customer":
        return {
            "tool_result": f"Found customer: {args.get('customer_name')}"
        }

    if tool_name == "send_email":
        print("\nEMAIL DETAILS")
        print(f"To: {args.get('to')}")
        print(f"Subject: {args.get('subject')}")
        print(f"Body: {args.get('body')}")

        return {
            "tool_result": (
                f"Email sent to {args.get('to')} "
                f"with subject '{args.get('subject')}'."
            )
        }

    return {
        "tool_result": f"Unknown tool: {tool_name}"
    }


# --------------------------------------------------
# 7. Rejected and blocked nodes
# --------------------------------------------------

def rejected_node(state: AgentState):
    print("\nREJECTED: Human rejected the action.")

    return {
        "tool_result": f"Action rejected. Reason: {state['human_feedback']}"
    }


def blocked_node(state: AgentState):
    print("\nBLOCKED: Middleware blocked the action.")

    return {
        "tool_result": f"Action blocked. Reason: {state['middleware_reason']}"
    }


# --------------------------------------------------
# 8. Routers
# --------------------------------------------------

def route_after_middleware(
    state: AgentState
) -> Literal["tool_node", "human_review_node", "blocked_node"]:
    decision = state["middleware_decision"]

    if decision == "allow":
        return "tool_node"

    if decision == "review":
        return "human_review_node"

    return "blocked_node"


def route_after_human_review(
    state: AgentState
) -> Literal["tool_node", "rejected_node"]:
    if state["human_decision"] in ["approve", "edit"]:
        return "tool_node"

    return "rejected_node"


# --------------------------------------------------
# 9. Build graph
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent_node", agent_node)
builder.add_node("middleware_node", middleware_node)
builder.add_node("human_review_node", human_review_node)
builder.add_node("tool_node", tool_node)
builder.add_node("rejected_node", rejected_node)
builder.add_node("blocked_node", blocked_node)

builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", "middleware_node")

builder.add_conditional_edges(
    "middleware_node",
    route_after_middleware
)

builder.add_conditional_edges(
    "human_review_node",
    route_after_human_review
)

builder.add_edge("tool_node", END)
builder.add_edge("rejected_node", END)
builder.add_edge("blocked_node", END)

graph = builder.compile(checkpointer=MemorySaver())

export_graph_image(graph, "middleware_hitl_graph.png")


# --------------------------------------------------
# 10. Human input helpers
# --------------------------------------------------

def ask_human_decision():
    while True:
        decision = input("\nType approve, edit, or reject: ").strip().lower()

        if decision in ["approve", "edit", "reject"]:
            return decision

        print("Invalid input. Please type: approve, edit, or reject")


def build_resume_value(decision: str):
    if decision == "approve":
        return {
            "decision": "approve",
            "feedback": ""
        }

    if decision == "reject":
        reason = input("Why do you reject it? ").strip()

        return {
            "decision": "reject",
            "feedback": reason
        }

    if decision == "edit":
        new_subject = input("New subject: ").strip()
        new_body = input("New body: ").strip()

        return {
            "decision": "edit",
            "feedback": "Human edited the email before sending.",
            "edited_args": {
                "to": "john@example.com",
                "subject": new_subject,
                "body": new_body
            }
        }

    raise ValueError("Invalid decision")


# --------------------------------------------------
# 11. Run demo
# --------------------------------------------------

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "pure-langgraph-middleware-hitl-demo"
        }
    }

    initial_state = {
        "user_request": "Send an email to John saying his refund is approved.",

        "proposed_tool": "",
        "proposed_args": {},

        "middleware_decision": "",
        "middleware_reason": "",

        "human_decision": "",
        "human_feedback": "",

        "tool_result": ""
    }

    print("\n--- Starting graph ---")

    result = graph.invoke(initial_state, config=config)

    print("\n--- Graph paused or finished ---")
    print(result)

    if "__interrupt__" in result:
        print("\n--- Middleware requested human review ---")
        print(result["__interrupt__"])

        decision = ask_human_decision()
        resume_value = build_resume_value(decision)

        print("\n--- Resuming graph ---")

        final_result = graph.invoke(
            Command(resume=resume_value),
            config=config
        )

        print("\n--- Final result ---")
        print(final_result)
    else:
        print("\n--- Final result ---")
        print(result)

    
