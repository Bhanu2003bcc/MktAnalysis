"""
graph/workflow.py — LangGraph multi-agent workflow orchestration
"""

from langgraph.graph import StateGraph, END
from loguru import logger

from graph.state import FinancialAgentState
from agents.market_researcher import market_researcher_node
from agents.data_analyst import data_analyst_node
from agents.report_writer import report_writer_node
from report.pdf_generator import pdf_generator_node


def route_after_researcher(state: FinancialAgentState) -> str:
    """Route to data_analyst or end on critical error."""
    if not state.get("stock_data") and len(state.get("errors", [])) > 2:
        logger.warning("Too many errors after researcher. Ending.")
        return END
    return "data_analyst"


def route_after_analyst(state: FinancialAgentState) -> str:
    if not state.get("analysis_data"):
        return END
    return "report_writer"


def route_after_writer(state: FinancialAgentState) -> str:
    if not state.get("report_data"):
        return END
    return "pdf_generator"


def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph workflow."""
    graph = StateGraph(FinancialAgentState)

    # Add agent nodes
    graph.add_node("market_researcher", market_researcher_node)
    graph.add_node("data_analyst",      data_analyst_node)
    graph.add_node("report_writer",     report_writer_node)
    graph.add_node("pdf_generator",     pdf_generator_node)

    # Entry point
    graph.set_entry_point("market_researcher")

    # Conditional edges with routing
    graph.add_conditional_edges("market_researcher", route_after_researcher,
                                 {"data_analyst": "data_analyst", END: END})
    graph.add_conditional_edges("data_analyst", route_after_analyst,
                                 {"report_writer": "report_writer", END: END})
    graph.add_conditional_edges("report_writer", route_after_writer,
                                 {"pdf_generator": "pdf_generator", END: END})

    graph.add_edge("pdf_generator", END)

    return graph.compile()


def run_analysis(
    ticker: str,
    start_date: str,
    end_date: str,
    company_name: str = "",
) -> FinancialAgentState:
    """
    Public entry point. Run the full multi-agent financial analysis.

    Returns the final state containing:
    - stock_data, news_data, analysis_data, report_data
    - messages (agent conversation log)
    - errors
    """
    initial_state: FinancialAgentState = {
        "ticker": ticker.upper().strip(),
        "company_name": company_name or ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "stock_data": None,
        "news_data": None,
        "analysis_data": None,
        "report_data": None,
        "messages": [],
        "current_agent": "market_researcher",
        "retry_count": 0,
        "errors": [],
        "completed": False,
    }

    logger.info(f"Starting analysis pipeline for {ticker}")
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    logger.info(f"Analysis pipeline complete for {ticker}")
    return final_state
