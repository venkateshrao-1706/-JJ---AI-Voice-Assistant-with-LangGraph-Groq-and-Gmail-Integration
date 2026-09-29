from dotenv import load_dotenv
from scipy.io.wavfile import write
from groq import Groq
import sounddevice as sd
import os
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain.tools import  tool 
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_google_credentials
)
import pyttsx3
import re

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

# 1. Sign in to Gmail with read-only permission
credentials = get_google_credentials(
    token_file="token.json",
    client_secrets_file="credentials.json",
    scopes=["https://www.googleapis.com/auth/gmail.readonly"]
)

def speak(text):
    """Safely cleans markdown text and speaks it out loud"""
    print(f"\nAssistant: {text}")
    try:
        # Remove markdown formatting symbols (*, _, `, #) so TTS reads smoothly
        clean_text = re.sub(r'[*`#_]', '', text)
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)  # Adjust speaking speed if needed
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")

# 2. Connect to Gmail
gmail_api = build_resource_service(credentials=credentials)

# 3. Get only search and reading tools
gmail_tools = [
    tool
    for tool in GmailToolkit(api_resource=gmail_api).get_tools()
    if tool.name in [
        "search_gmail",
        "get_gmail_message",
        "get_gmail_thread"
    ]
]

print("Gmail connected successfully")

duration = 8
samplerate = 16000
filename = "WOHOO.wav"

search_tool = DuckDuckGoSearchRun()
search_tool.name = "web_search"
search_tool.description = ("Use this tool to search the live web for current events, recent news, real-time weather, "
    "or up-to-date facts that are not available in a static encyclopedia.")

api_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)

@tool
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for historical facts, scientific concepts,
    biographies, and general encyclopedic knowledge.
    """

    try:
        return api_wiki.run(query)

    except Exception as error:
        return f"Wikipedia search failed: {error}"


all_tolls = [search_tool,wikipedia_tool,*gmail_tools]

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


agent = create_agent(
    model=llm,
    tools= all_tolls,
    system_prompt=
    "You are an intelligent assistant with access to web search,"
        "Wikipedia, and read-only Gmail tools. "

        "Answer simple questions directly. "
        "Use web_search for recent or current information. "
        "Use wikipedia_tool for encyclopedic information. "
        "Use Gmail tools only when the user asks about their emails. "
        "You cannot send, delete, or modify emails."
)

while True:
    try:
        print("-----------------------------------------")
        print("Speak Now........")

        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="int16"
        )
        sd.wait()

        write(filename, samplerate, audio)

        print("Listening user voice.............................")
        print("Gathering user spoken info............")
        print("Processing your query. Hold on...............")

        with open(filename, 'rb') as audiofile:
            user_input = client.audio.transcriptions.create(
                file=(filename, audiofile.read()),
                model="whisper-large-v3-turbo",
                language='en'
            )
        
        spoken_text = user_input.text.strip()
        print(f"You Said: {spoken_text}")

        # Exit condition if user says quit or exit
        if any(keyword in spoken_text.lower() for keyword in ["exit", "quit", "stop"]):
            print("\n👋 Goodbye! Shutting down assistant.")
            break

        response = agent.invoke({"messages": [{"role": "user","content": spoken_text}]})

        assistant_answer = response["messages"][-1].content

        print(f"\nAssistant: {assistant_answer}\n")

        speak(assistant_answer)

    except KeyboardInterrupt:
        print("\n👋 Assistant stopped manually. Goodbye!")
        break
    except Exception as e:
        print(f"\n⚠️ An error occurred: {e}")
        print("Let's try again...\n")