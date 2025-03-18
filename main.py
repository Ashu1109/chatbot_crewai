from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn
from crewai import Agent, Task, Crew, Process, LLM

app = FastAPI()

# LLM setup
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)

class AgentResponse(BaseModel):
    result: str

def create_agents():
    guide = Agent(
        role="Senior Guide for Event Management Services",
        goal="Provide detailed guide on event management services in India",
        backstory="10+ years experience in Indian event management industry",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    
    manager = Agent(
        role="Event Manager",
        goal="Manage event services in India",
        backstory="5+ years experience managing events in India",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    return guide, manager

@app.options("/get-event-management-guide", response_model=AgentResponse)
async def get_event_management_guide():
    guide, manager = create_agents()
    
    task1 = Task(
        description="Provide detailed guide on event management services in India",
        expected_output="Comprehensive guide including services, processes, and value propositions",
        agent=guide
    )

    task2 = Task(
        description="Review and manage the event management services guide",
        expected_output="Reviewed and managed guide for event management services",
        agent=manager
    )

    crew = Crew(
        agents=[guide, manager],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    # Convert CrewOutput to string using the raw attribute
    return AgentResponse(result=result.raw)

@app.post("/health")
def gets():
    return AgentResponse(result="hello")

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
