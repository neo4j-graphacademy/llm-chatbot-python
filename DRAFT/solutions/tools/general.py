# tag::importtool[]
from langchain.tools import Tool
# end::importtool[]

from solutions import llm

# tag::tool[]
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    )
]
# end::tool[]