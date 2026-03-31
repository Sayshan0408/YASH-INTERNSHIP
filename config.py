"""
agents/config.py
Pipeline constants and agent metadata.
Model: llama-3.1-8b-instant (Groq production model — fast, stable, not deprecated)
"""

# Execution order of all 6 agents
STAGE_ORDER = [
    "planner",
    "architect",
    "developer",
    "reviewer",
    "tester",
    "documenter",
]

# Display metadata for the UI
AGENT_META = {
    "planner": {
        "label": "Planner",
        "icon": "📋",
        "color": "#3B82F6",
        "description": "Breaks the requirement into tasks, milestones, and recommends a tech stack.",
    },
    "architect": {
        "label": "Architect",
        "icon": "🏗️",
        "color": "#8B5CF6",
        "description": "Designs system architecture, folder structure, and key design patterns.",
    },
    "developer": {
        "label": "Developer",
        "icon": "💻",
        "color": "#10B981",
        "description": "Writes complete, production-ready code based on the plan and architecture.",
    },
    "reviewer": {
        "label": "Reviewer",
        "icon": "🔍",
        "color": "#F59E0B",
        "description": "Reviews code for bugs, security issues, and best practices.",
    },
    "tester": {
        "label": "Tester",
        "icon": "🧪",
        "color": "#EF4444",
        "description": "Writes unit tests and identifies edge cases.",
    },
    "documenter": {
        "label": "Documenter",
        "icon": "📝",
        "color": "#06B6D4",
        "description": "Generates README, API docs, and usage examples.",
    },
}

# Groq model — llama-3.1-8b-instant is the current stable fast production model
# Alternative (higher quality, more tokens): "llama-3.3-70b-versatile"
GROQ_MODEL = "llama-3.1-8b-instant"

# Developer <-> Reviewer loop settings
MAX_REVIEW_ITERATIONS = 3
REVIEW_PASS_KEYWORD = "LGTM"
