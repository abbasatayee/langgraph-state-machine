import os
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# LangGraph versions may differ slightly.
# Newer versions use InMemorySaver.
# Older versions may use MemorySaver.
try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver


load_dotenv()


# ============================================================
# 1. State
# ============================================================

class SupportState(TypedDict):
    user_message: str

    intent: Optional[Literal["general_chat", "support_question", "unclear"]]

    kb_result: Optional[str]
    draft_answer: Optional[str]

    answer_quality: Optional[Literal["answer_good", "not_enough"]]

    human_approval: Optional[Literal["approved", "rejected"]]

    ticket_id: Optional[str]
    final_response: Optional[str]


# ============================================================
# 2. Model
# ============================================================

model_name = os.getenv("MODEL_NAME", "ollama:minimax-m2.5:cloud")

model = init_chat_model(
    model_name,
    temperature=0,
)


# ============================================================
# 3. Fake Knowledge Base
# ============================================================

KNOWLEDGE_BASE = {
    "refund": (
        "Refunds are available within 14 days of purchase. "
        "The product must not be heavily used. Refunds usually take 5-10 business days."
    ),
    "password": (
        "Users can reset their password from the login page by clicking 'Forgot password'. "
        "A reset link will be sent to their registered email."
    ),
    "delivery": (
        "Standard delivery takes 3-5 business days. "
        "Express delivery takes 1-2 business days."
    ),
}


# ============================================================
# 4. Nodes
# ============================================================

def classify_intent(state: SupportState) -> dict:
    """
    Decide what type of user request this is.
    """

    response = model.invoke([
        SystemMessage(content="""
You classify customer support messages.

Return only one of these labels:
- general_chat
- support_question
- unclear

Rules:
- If the user asks about refund, password, delivery, account, payment, charge, billing, product issue, or technical issue, return support_question.
- If the user is greeting or asking casual/general things, return general_chat.
- If the message is too vague, return unclear.
"""),
        HumanMessage(content=state["user_message"]),
    ])

    intent = response.content.strip().lower()

    if intent not in ["general_chat", "support_question", "unclear"]:
        intent = "unclear"

    return {
        "intent": intent
    }


def respond_directly(state: SupportState) -> dict:
    """
    Answer simple/general messages directly.
    """

    response = model.invoke([
        SystemMessage(content="You are a friendly customer support assistant. Be concise."),
        HumanMessage(content=state["user_message"]),
    ])

    return {
        "final_response": response.content
    }


def ask_clarification(state: SupportState) -> dict:
    """
    Ask the user for more information.
    """

    return {
        "final_response": (
            "Could you please clarify your request? "
            "For example, are you asking about a refund, delivery, password reset, billing, or another issue?"
        )
    }


def search_knowledge_base(state: SupportState) -> dict:
    """
    Search the fake knowledge base.
    In real production, this could be Pinecone, Elasticsearch, Postgres, Zendesk, etc.
    """

    user_message = state["user_message"].lower()

    matched_articles = []

    for keyword, article in KNOWLEDGE_BASE.items():
        if keyword in user_message:
            matched_articles.append(article)

    if not matched_articles:
        return {
            "kb_result": "No relevant knowledge base article found."
        }

    return {
        "kb_result": "\n\n".join(matched_articles)
    }


def generate_answer(state: SupportState) -> dict:
    """
    Generate an answer using the KB result.
    """

    response = model.invoke([
        SystemMessage(content="""
You are a customer support assistant.

Answer the user using only the provided knowledge base result.
If the knowledge base does not contain enough information,
say that the issue may need a support ticket.
Be concise and helpful.
"""),
        HumanMessage(content=f"""
User question:
{state["user_message"]}

Knowledge base result:
{state["kb_result"]}
"""),
    ])

    return {
        "draft_answer": response.content
    }


def evaluate_answer(state: SupportState) -> dict:
    """
    Decide whether the generated answer is enough.
    """

    # Deterministic shortcut:
    # if KB found nothing, we already know the answer is not enough.
    if state["kb_result"] == "No relevant knowledge base article found.":
        return {
            "answer_quality": "not_enough"
        }

    response = model.invoke([
        SystemMessage(content="""
You evaluate customer support answers.

Return only one of:
- answer_good
- not_enough

Return answer_good if the draft answer directly answers the user's question.
Return not_enough if the answer is uncertain, incomplete, or says a ticket may be needed.
"""),
        HumanMessage(content=f"""
User question:
{state["user_message"]}

Knowledge base result:
{state["kb_result"]}

Draft answer:
{state["draft_answer"]}
"""),
    ])

    quality = response.content.strip().lower()

    if quality not in ["answer_good", "not_enough"]:
        quality = "not_enough"

    return {
        "answer_quality": quality
    }


def finish_with_answer(state: SupportState) -> dict:
    """
    Finish with the generated answer.
    """

    return {
        "final_response": state["draft_answer"]
    }


def request_human_approval(state: SupportState) -> dict:
    """
    Pause the graph and wait for human approval before creating a ticket.
    """

    approval = interrupt({
        "question": "Do you approve creating a support ticket?",
        "user_message": state["user_message"],
        "kb_result": state["kb_result"],
        "draft_answer": state["draft_answer"],
        "reason": "The knowledge base answer was not enough.",
        "allowed_responses": ["approved", "rejected"],
    })

    approval = str(approval).strip().lower()

    if approval not in ["approved", "rejected"]:
        approval = "rejected"

    return {
        "human_approval": approval
    }


def create_ticket(state: SupportState) -> dict:
    """
    Create a support ticket.
    In real life, this could call Zendesk, Jira, HubSpot, Freshdesk, etc.
    """

    fake_ticket_id = "TICKET-1001"

    return {
        "ticket_id": fake_ticket_id,
        "final_response": (
            "I could not fully answer this from the available knowledge base, "
            f"so a support ticket has been created. Your ticket ID is {fake_ticket_id}."
        )
    }


def do_not_create_ticket(state: SupportState) -> dict:
    """
    Handle rejected approval.
    """

    return {
        "final_response": (
            "I could not fully answer this from the available knowledge base, "
            "and ticket creation was not approved. "
            "Please provide more details so I can help further."
        )
    }


# ============================================================
# 5. Routers / Conditional Edge Functions
# ============================================================

def route_intent(state: SupportState) -> str:
    """
    Route after intent classification.
    """

    if state["intent"] == "general_chat":
        return "respond_directly"

    if state["intent"] == "support_question":
        return "search_knowledge_base"

    return "ask_clarification"


def route_after_evaluation(state: SupportState) -> str:
    """
    Route after checking answer quality.
    """

    if state["answer_quality"] == "answer_good":
        return "finish_with_answer"

    return "request_human_approval"


def route_after_human_approval(state: SupportState) -> str:
    """
    Route after human approval.
    """

    if state["human_approval"] == "approved":
        return "create_ticket"

    return "do_not_create_ticket"


# ============================================================
# 6. Build Graph
# ============================================================

graph_builder = StateGraph(SupportState)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("respond_directly", respond_directly)
graph_builder.add_node("ask_clarification", ask_clarification)
graph_builder.add_node("search_knowledge_base", search_knowledge_base)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("evaluate_answer", evaluate_answer)
graph_builder.add_node("finish_with_answer", finish_with_answer)
graph_builder.add_node("request_human_approval", request_human_approval)
graph_builder.add_node("create_ticket", create_ticket)
graph_builder.add_node("do_not_create_ticket", do_not_create_ticket)

graph_builder.add_edge(START, "classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "respond_directly": "respond_directly",
        "search_knowledge_base": "search_knowledge_base",
        "ask_clarification": "ask_clarification",
    },
)

graph_builder.add_edge("respond_directly", END)
graph_builder.add_edge("ask_clarification", END)

graph_builder.add_edge("search_knowledge_base", "generate_answer")
graph_builder.add_edge("generate_answer", "evaluate_answer")

graph_builder.add_conditional_edges(
    "evaluate_answer",
    route_after_evaluation,
    {
        "finish_with_answer": "finish_with_answer",
        "request_human_approval": "request_human_approval",
    },
)

graph_builder.add_edge("finish_with_answer", END)

graph_builder.add_conditional_edges(
    "request_human_approval",
    route_after_human_approval,
    {
        "create_ticket": "create_ticket",
        "do_not_create_ticket": "do_not_create_ticket",
    },
)

graph_builder.add_edge("create_ticket", END)
graph_builder.add_edge("do_not_create_ticket", END)

checkpointer = InMemorySaver()

app = graph_builder.compile(checkpointer=checkpointer)


# ============================================================
# 7. Helper Functions
# ============================================================

def create_initial_state(user_message: str) -> SupportState:
    return {
        "user_message": user_message,
        "intent": None,
        "kb_result": None,
        "draft_answer": None,
        "answer_quality": None,
        "human_approval": None,
        "ticket_id": None,
        "final_response": None,
    }


def has_interrupt(result) -> bool:
    """
    Handles different LangGraph result shapes across versions.
    """

    if isinstance(result, dict) and "__interrupt__" in result:
        return True

    if hasattr(result, "interrupts") and result.interrupts:
        return True

    return False


def print_interrupt(result) -> None:
    """
    Print interrupt payload in a friendly way.
    """

    print("\n--- HUMAN APPROVAL REQUIRED ---")

    if isinstance(result, dict) and "__interrupt__" in result:
        interrupts = result["__interrupt__"]

        for item in interrupts:
            value = getattr(item, "value", item)
            print(value)

        return

    if hasattr(result, "interrupts"):
        for item in result.interrupts:
            value = getattr(item, "value", item)
            print(value)


def print_final_result(result) -> None:
    """
    Print important result fields.
    """

    print("\n--- FINAL RESULT ---")
    print("Intent:", result.get("intent"))
    print("KB result:", result.get("kb_result"))
    print("Answer quality:", result.get("answer_quality"))
    print("Human approval:", result.get("human_approval"))
    print("Ticket ID:", result.get("ticket_id"))
    print("Final response:", result.get("final_response"))


# ============================================================
# 8. Run CLI Test
# ============================================================

if __name__ == "__main__":
    print("\nCustomer Support Agent with Human Approval")
    print("=" * 60)

    print("\nTry one of these:")
    print("- Hi, how are you?")
    print("- How can I reset my password?")
    print("- I want my money back. What is your refund policy?")
    print("- My package has not arrived yet. How long does delivery take?")
    print("- My account was charged twice yesterday.")
    print("- I have a problem.")

    user_input = input("\nUser: ").strip()

    if not user_input:
        user_input = "My account was charged twice yesterday."

    config = {
        "configurable": {
            "thread_id": "support-agent-demo-1"
        }
    }

    result = app.invoke(
        create_initial_state(user_input),
        config=config,
    )

    if has_interrupt(result):
        print_interrupt(result)

        approval = input("\nType 'approved' or 'rejected': ").strip().lower()

        if approval not in ["approved", "rejected"]:
            approval = "rejected"

        result = app.invoke(
            Command(resume=approval),
            config=config,
        )

    print_final_result(result)