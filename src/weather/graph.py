import uuid
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer, push_ui_message


class AgentState(TypedDict):  # noqa: D101
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]


class Weather(TypedDict):
    temperature: float
    conditions: str
    humidity: float
    wind_speed: float


class WeatherOutput(TypedDict):
    city: str
    weather: Weather


async def weather(state: AgentState):
    weather: WeatherOutput = (
        await ChatOpenAI(model="gpt-4o-mini")
        .with_structured_output(WeatherOutput)
        .with_config({"tags": ["nostream"]})
        .ainvoke(state["messages"])
    )

    message = AIMessage(
        id=str(uuid.uuid4()),
        content=f"Here's the weather for {weather['city']}",
    )

    # Emit UI elements associated with the message
    push_ui_message("weatherComponent", weather, message=message)
    return {"messages": [message]}


workflow = StateGraph(AgentState)
workflow.add_node(weather)
workflow.add_edge("__start__", "weather")
graph = workflow.compile(name="WeatherAgent")
