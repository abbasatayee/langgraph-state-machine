import os
import operator
from dotenv import load_dotenv
from pyexpat.errors import messages
from langchain_core.tools import tool
from typing import Annotated , TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    tool_calls: int


# -----------------------------
# 2. Create a simple tool
# -----------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake data for learning.
    # Later, this could call a real weather API.
    weather_data = {
        "berlin": "The weather in Berlin is cloudy and 18°C.",
        "kabul": "The weather in Kabul is sunny and 24°C.",
        "hamburg": "The weather in Hamburg is rainy and 16°C.",
    }

    city_key = city.lower()

    return weather_data.get(
        city_key,
        f"I do not have weather data for {city}."
    )


tools = [get_weather]
tools_by_name = {tool.name: tool for tool in tools}


model_name = os.getenv("MODEL_NAME", "ollama:minimax-m2.5:cloud")

model = init_chat_model(
    model_name,
    temperature=0,
)

model_with_tools = model.bind_tools(tools)


# -----------------------------
# 4. Node: AI thinks
# -----------------------------
def ai_thinks(state: AgentState) -> AgentState:
    """
    This node calls the LLM.

    The LLM can either:
    1. Answer directly
    2. Request a tool call
    """

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "If the user asks about weather, use the get_weather tool. "
            "If you already have the tool result, answer the user clearly."
        )
    )

    response = model_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response],
        "tool_calls": 0,
    }

# -----------------------------
# 5. Conditional edge: decide next step
# -----------------------------
def should_continue(state: AgentState) -> str:
    """
    Decide where the graph should go next.

    If the latest AI message has tool calls:
        go to execute_tool

    Otherwise:
        end the graph
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "execute_tool"

    return "end"

# -----------------------------
# 6. Node: execute tool
# -----------------------------
def execute_tool(state: AgentState) -> AgentState:
    """
    This node executes the tool requested by the AI.

    Example:
    AI says: call get_weather(city="Berlin")
    This node actually runs get_weather("Berlin")
    """

    last_message = state["messages"][-1]

    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        selected_tool = tools_by_name[tool_name]
        print(f"Executing tool: {tool_name} with args: {tool_args}")

        tool_result = selected_tool.invoke(tool_args)

        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call_id,
        )

        tool_messages.append(tool_message)

    return {
        "messages": tool_messages,
        "tool_calls": state["tool_calls"] + len(tool_messages),
    }

def build_agent_graph():
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("ai_thinks", ai_thinks)
    graph_builder.add_node("execute_tool", execute_tool)

    # Start here
    graph_builder.add_edge(START, "ai_thinks")
    # Conditional edge:
    # after ai_thinks, decide what happens next
    graph_builder.add_conditional_edges(
        "ai_thinks",
        should_continue,
        {
            "execute_tool": "execute_tool",
            "end": END,
        },
    )

    graph_builder.add_edge("execute_tool", "ai_thinks")
    return graph_builder.compile()


def run_ai_agent():
    print("Running AI Agent...")
    app = build_agent_graph()
    result = app.invoke({
        "messages": [
            HumanMessage(content="What is the weather in Berlin?")
        ],
        "tool_calls": 0,
    })

    print("\nFinal messages:")
    for message in result["messages"]:
        print(f"\n{message.type.upper()}:")
        print(message.content)

    print("\nTotal tool calls:", result["tool_calls"])


if __name__ == "__main__": 
    run_ai_agent()
