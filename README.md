🎓 CampusAI Navigator

AI-powered campus assistant for college students
Groq · Llama 3.3 70B · ReAct Agent · Semantic Chunking · Streamlit
📌 What Is This?
CampusAI Navigator is a conversational AI assistant built for college students. Ask anything about faculty, events, clubs, internships, resources, and more. It uses a ReAct (Reasoning + Acting) agent loop to search the web, fetch and semantically chunk pages, and deliver cited, structured answers — all powered by Llama 3.3 70B on Groq.

🔄 How It Works
User Query
    │
    ▼
System Prompt Builder
(college + mode + search hint + date)
    │
    ▼
ReAct Agent Loop (max 5 iterations)
    │
    ├── <thought>  →  reasoning step
    ├── <action>   →  search_web: query
    │                  OR fetch_page: url
    │                      │
    │                 Semantic Chunking
    │                 + Chunk Ranking
    │                 (keyword / semantic / hybrid)
    └── <final_answer>  →  Markdown + Sources
    │
    ▼
Streamlit UI
(chat bubbles · source pills · ReAct trace expander)

🎯 Focus Modes
ModeContext🎓 GeneralClubs, administration, campus life🎉 EventsHackathons, fests, seminars, workshops👩‍🏫 FacultyProfessors, research, office hours, contacts📚 ResourcesLibraries, labs, scholarships, mental health💼 InternshipsPlacements, recruiters, career tips

🔍 Search Modes
ModeAlgorithmBest For🔑 KeywordToken overlap scoringNames, codes, exact terms🧠 SemanticTF-IDF cosine similarityTopics, themes, concepts⚡ Hybrid (default)50% keyword + 50% semanticGeneral use

🧩 Key Components
semantic_chunk(text)
Splits raw page text into overlapping sentence-boundary chunks (~400 words, 80-word overlap) for coherent retrieval.
rank_chunks(chunks, query, search_mode)
Scores every chunk against the query using the selected algorithm. Returns top-k chunks sorted by original document order to preserve context flow.
run_agent(query, ...)
The core ReAct loop — calls Groq, parses <thought>, <action>, <final_answer> tags, dispatches tools, appends observations, and loops up to 5 iterations. Falls back to a synthesis call if no <final_answer> is produced.
parse_tags(text) + parse_action(s)
Zero-dependency regex tag parser — no function-calling APIs needed. Works entirely on raw model text output.

⚙️ Sampling Parameters
PhaseTemperatureTop-pReAct reasoning0.20.9Final answer0.70.9
Top-k chunks per fetch: 4 · Max iterations: 5 · Max tokens: 4096

🚀 Quick Start
bashgit clone https://github.com/your-username/campusai-navigator.git
cd campusai-navigator
pip install -r requirements.txt
Add your Groq API key:
bashecho "GROQ_API_KEY=gsk_your_key_here" > .env
Run:
bashstreamlit run app.py
Get a free Groq key at 👉 console.groq.com/keys

📦 Requirements
streamlit
requests
beautifulsoup4
python-dotenv

🗂️ Project Structure
campusai-navigator/
│
├── app.py                  # Full application
│   ├── Constants & Config
│   ├── CSS Styles          # Navy/gold dark theme
│   ├── Search Algorithms   # keyword / semantic / hybrid
│   ├── Web Tools           # search_web(), fetch_page()
│   ├── Groq API Client     # groq_chat(), build_system()
│   ├── Tag Parser          # parse_tags(), parse_action()
│   ├── ReAct Agent         # run_agent()
│   └── Streamlit UI        # sidebar, chat, search buttons
│
├── .env                    # API key (not committed)
├── requirements.txt
└── README.md

💡 Example Questions
"Which professor is best for ML thesis guidance?"
"What hackathons are coming up this month?"
"Where can I get free software licenses?"
"How do I register with the placement cell?"
"What mental health resources are available on campus?"
