import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class IncidentRequest(BaseModel):
    incident: str

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for live disaster news and emergency updates",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather alerts for a location",
            "parameters": {
                "type": "object",
                "properties": {"state_code": {"type": "string"}},
                "required": ["state_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_resources",
            "description": "Get available shelters, hospitals, and road status",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }
]

def web_search(query):
    results = tavily.search(query=query, max_results=3)
    return str([r["content"] for r in results["results"]])

def get_weather(state_code):
    import requests
    r = requests.get(
        f"https://api.weather.gov/alerts/active?area={state_code}",
        headers={"User-Agent": "ARIA-DisasterAgent"}
    )
    data = r.json()
    alerts = data.get("features", [])[:2]
    if not alerts:
        return "No active weather alerts found."
    return str([a["properties"]["headline"] for a in alerts])

def get_resources(location):
    return json.dumps({
        "shelters": [
            {"name": "Convention Center", "capacity": 2000, "available": 800},
            {"name": "High School Gym", "capacity": 500, "available": 200}
        ],
        "hospitals": [
            {"name": "Tampa General", "trauma_level": 1, "er_wait": "45min"},
            {"name": "St. Joseph's Hospital", "trauma_level": 2, "er_wait": "20min"}
        ],
        "roads": {
            "I-75": "open",
            "I-4": "closed - flooding",
            "US-19": "congested"
        }
    })

def run_tool(name, args):
    if name == "web_search":
        return web_search(args["query"])
    if name == "get_weather":
        return get_weather(args["state_code"])
    if name == "get_resources":
        return get_resources(args["location"])

RECON_PROMPT = """You are the Recon Agent for ARIA, a disaster response system.
Your job is to gather live situational data about the incident using your tools.
Call web_search, get_weather, and get_resources to collect as much real data as possible.
Think out loud before each action:
REASONING: [why you are calling this tool]
ACTION: [tool you are calling]
OBSERVATION: [what you found]
After gathering data, summarize everything you found."""

ANALYSIS_PROMPT = """You are the Analysis Agent for ARIA, a disaster response system.
You will receive raw situational data from the Recon Agent.
Your job is to:
1. Identify the top 3 immediate threats ranked by urgency
2. Perform a gap analysis: what resources are needed vs available
3. Identify the most vulnerable populations at risk
Be specific and data-driven. Think step by step."""

COMMANDER_PROMPT = """You are the Commander Agent for ARIA, a disaster response system.
You will receive a threat analysis and must produce a structured response plan.
Output exactly in this format:

INCIDENT BRIEFING
- Location:
- Disaster type:
- Severity:
- Estimated affected population:
- Time-critical window:

SITUATION SUMMARY
[2-3 sentences]

IMMEDIATE THREATS
1. [Threat] - [Why urgent] - [Time window]
2.
3.

RESOURCE STATUS
- Shelters:
- Medical:
- Transport:

PRIORITY ACTION PLAN
[ ] CRITICAL (0-2 hrs):
[ ] URGENT (2-6 hrs):
[ ] IMPORTANT (6-24 hrs):

DECISIONS REQUIRED BY HUMAN COORDINATOR
1.
2.
3.

NEXT REASSESSMENT: [when and what trigger]"""

def run_agent_loop(system_prompt, user_message, model, use_tools=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    while True:
        kwargs = {"model": model, "messages": messages, "max_tokens": 2048, "temperature": 0.6}
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        if not use_tools or not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

def aria_stream(incident: str):
    yield f"data: [RECON] Gathering live situational data...\n\n"
    recon_output = run_agent_loop(
        RECON_PROMPT, incident,
        model="nvidia/nemotron-3-nano-30b-a3b",
        use_tools=True
    )
    for line in recon_output.split("\n"):
        if line.strip():
            yield f"data: [RECON] {line}\n\n"

    yield f"data: [ANALYSIS] Analyzing threats and resource gaps...\n\n"
    analysis_output = run_agent_loop(
        ANALYSIS_PROMPT,
        f"Incident: {incident}\n\nRecon Data:\n{recon_output}",
        model="nvidia/nemotron-3-super-120b-a12b",
        use_tools=False
    )
    for line in analysis_output.split("\n"):
        if line.strip():
            yield f"data: [ANALYSIS] {line}\n\n"

    yield f"data: [COMMANDER] Generating response plan...\n\n"
    final_plan = run_agent_loop(
        COMMANDER_PROMPT,
        f"Incident: {incident}\n\nAnalysis:\n{analysis_output}",
        model="nvidia/nemotron-3-super-120b-a12b",
        use_tools=False
    )
    for line in final_plan.split("\n"):
        if line.strip():
            yield f"data: [COMMANDER] {line}\n\n"

    yield f"data: [DONE] {json.dumps({'plan': final_plan})}\n\n"

@app.post("/run-aria")
async def run_aria(request: IncidentRequest):
    return StreamingResponse(
        aria_stream(request.incident),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health():
    return {"status": "ARIA is online"}