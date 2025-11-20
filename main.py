#!/usr/bin/env python3
"""
Interactive example agent for agentic_ai_talk.

Usage:
    python main.py
"""
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment (.env)")

# LangChain imports
try:
    from langchain import OpenAI
    from langchain.agents import initialize_agent, Tool, AgentType
except Exception as e:
    raise RuntimeError(
        "Missing dependencies. Run: pip install -r requirements.txt"
    ) from e

# --- Tools -------------------------------------------------
def calculator_tool(input_str: str) -> str:
    """
    Very small safe evaluator for arithmetic expressions.
    For demo purposes only. DO NOT use eval on untrusted input in production.
    """
    allowed = "0123456789+-*/(). "
    if any(ch not in allowed for ch in input_str):
        return "Error: only simple arithmetic characters are allowed."
    try:
        # Safe eval: block builtins
        result = eval(input_str, {"__builtins__": None}, {})
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"

def echo_tool(input_str: str) -> str:
    """Simple echo tool to demonstrate custom tools"""
    return f"ECHO: {input_str}"

# Wrap tools for LangChain
tools = [
    Tool(name="Calculator", func=calculator_tool, description="Evaluates simple arithmetic expressions."),
    Tool(name="Echo", func=echo_tool, description="Returns the input prefixed with ECHO."),
]

# --- Agent setup -----------------------------------------
def build_agent():
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

# --- CLI loop --------------------------------------------
def cli_loop():
    agent = build_agent()
    print("Agent ready. Type 'exit' or Ctrl-C to quit.")
    while True:
        try:
            task = input("\nTask: ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            output = agent.run(task)
            print("\nAgent result:\n", output)
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    cli_loop()
