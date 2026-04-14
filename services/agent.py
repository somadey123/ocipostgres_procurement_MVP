import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_oci.chat_models import ChatOCIGenAI

from core.config import oci_auth_file_location
from services.tools import search_procurement_db, search_procurement_policy


SYSTEM_PROMPT = """
You are a procurement assistant MVP.
Rules:
- Ask what the user wants to procure if the request is unclear.
- Use the search_procurement_db tool to check inventory and preferred vendors.
- Use the search_procurement_policy tool to check policy.
- Ground your response in tool output only.
- If search_procurement_db returns one or more inventory rows, you MUST list them as matched inventory.
- Do not say "no matching inventory" unless the inventory array is actually empty.
- Do not say "no preferred vendors" unless the vendors array is actually empty.
- Never claim you placed an order.
- Return a concise recommendation with:
  1) matched inventory
  2) preferred vendors
  3) policy notes
  4) confidence
  5) next step
If evidence is weak, ask a clarification question.
"""


def get_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOCIGenAI(
        model_id=os.environ["OCI_MODEL_ID"],
        service_endpoint=os.environ["OCI_ENDPOINT"],
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
        model_kwargs={"temperature": 0.0, "max_tokens": 800},
        auth_profile=os.environ.get("OCI_PROFILE", "DEFAULT"),
        auth_file_location=oci_auth_file_location(),
    )

    tools = [search_procurement_db, search_procurement_policy]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
