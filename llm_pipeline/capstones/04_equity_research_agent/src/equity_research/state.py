from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator


SubAgent = Literal["fundamentals", "news", "filings", "finalise"]


class ResearchState(TypedDict):
    messages: Annotated[list[str], operator.add]
    ticker: str
    question: str
    next_step: SubAgent | None     # supervisor decision
    fundamentals_notes: str
    news_notes: str
    filings_notes: str
    draft_report: str
    approved: bool
