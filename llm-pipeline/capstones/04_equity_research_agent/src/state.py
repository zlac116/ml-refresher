from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


SubAgent = Literal["fundamentals", "news", "filings", "finalise"]


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticker: str
    question: str
    next_step: SubAgent | None     # supervisor decision
    fundamentals_notes: str
    news_notes: str
    filings_notes: str
    draft_report: str
    approved: bool
