import os
from typing import Optional
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
search_tool = SerperDevTool()

# Initialize LLM with environment variable
llm = LLM(
    model=("groq/gemma2-9b-it"),
    api_key=os.getenv("GROQ_API_KEY"),
)


# Define Agents
guide = Agent(
    role="Senior Guide for Event Management Services in India",
    goal="Provide detailed information about event management services in India",
    backstory="Experienced professional with 10+ years in Indian event management industry",
    llm=llm,
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
)

manager = Agent(
    role="Quality Assurance Manager",
    goal="Ensure the output is concise, accurate and engaging",
    backstory="Detail-oriented professional with 5+ years refining content for event management",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# FastAPI setup
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventRequest(BaseModel):
    query: str
    max_words: Optional[int] = 100  # Add configurable word limit


@app.get("/")
async def read_root():
    return {"status": "active", "service": "Event Management AI"}


@app.post("/get-event-info")
async def get_event_info(request: EventRequest):
    try:
        # Create tasks
        research_task = Task(
            description=f"Research and provide comprehensive information about: {request.query}",
            expected_output="Detailed factual information about the requested event service",
            agent=guide,
        )

        refine_task = Task(
            description=f"Refine the information to be concise and engaging (max {request.max_words} words)",
            expected_output="Polished, accurate content ready for client delivery",
            agent=manager,
        )

        # Execute crew
        crew = Crew(
            agents=[guide, manager],
            tasks=[research_task, refine_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        print(result)

        return {"response": result, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
