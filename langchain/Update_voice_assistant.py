import os
import time
import json
import uuid
import logging
import re

from datetime import datetime, timezone
from dotenv import load_dotenv

import speech_recognition as sr
import pyttsx3
import jiwer
import psycopg

from psycopg.rows import dict_row
from groq import Groq

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_google_credentials
)
from langgraph.checkpoint.postgres import PostgresSaver


# --------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logger = logging.getLogger("voice_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.FileHandler("assistant.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log_event(request_id, component, status, **kwargs):
    event = {
        "time_stamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "component": component,
        "status": status,
        **kwargs
    }
    logger.info(json.dumps(event))


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

def speak(text):
    print(f"\nJJ: {text}")

    try:
        clean_text = re.sub(r'[*`#_]', '', str(text))

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()

    except Exception as e:
        print(f"TTS Error: {e}")


# --------------------------------------------------
# GMAIL CONNECTION
# --------------------------------------------------

credentials = get_google_credentials(
    token_file="token.json",
    client_secrets_file="C:/Users/venka/OneDrive/Documents/RERSUME FOR VENKATESH/credentials.json",
    scopes=["https://www.googleapis.com/auth/gmail.readonly"]
)

gmail_api = build_resource_service(credentials=credentials)

gmail_tools = [
    gmail_tool
    for gmail_tool in GmailToolkit(api_resource=gmail_api).get_tools()
    if gmail_tool.name in [
        "search_gmail",
        "get_gmail_message",
        "get_gmail_thread"
    ]
]

print("Gmail connected successfully!")


# --------------------------------------------------
# AUDIO SETTINGS
# --------------------------------------------------

filename = "WOHOO.wav"
recognizer = sr.Recognizer()
recognizer.pause_threshold = 3.0

START_TIMEOUT = 10
MAX_RECORDING = 30
DEBUG_MODE = False

wake_words = ["hey jj", "hey jay jay", "hey j j"]
assistant_awake = False


# --------------------------------------------------
# SEARCH TOOLS
# --------------------------------------------------

search_tool = DuckDuckGoSearchRun()
search_tool.name = "web_search"
search_tool.description = (
    "Use this tool to search the live web for current events, "
    "recent news, real-time weather, or up-to-date facts."
)

api_wiki = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=500
)


@tool
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for historical facts, scientific concepts,
    biographies, and general encyclopedic knowledge."""
    try:
        return api_wiki.run(query)
    except Exception as error:
        return f"Wikipedia search failed: {error}"


all_tools = [search_tool, wikipedia_tool, *gmail_tools]


# --------------------------------------------------
# LANGUAGE MODEL
# --------------------------------------------------

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# --------------------------------------------------
# POSTGRESQL MEMORY
# --------------------------------------------------

connection = psycopg.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    autocommit=True,
    row_factory=dict_row
)

memory = PostgresSaver(connection)
memory.setup()
print("PostgreSQL memory connected successfully!")


# --------------------------------------------------
# LANGGRAPH AGENT
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=all_tools,
    checkpointer=memory,
    system_prompt=(
        "You are JJ, a friendly, natural-sounding voice assistant. "
        "Speak conversationally, like a helpful person rather than a robot. "
        "Keep spoken answers reasonably concise unless the user asks for detail. "
        "You have access to web search, Wikipedia, and read-only Gmail tools. "
        "Answer simple questions directly. "
        "Use web_search for recent or current information. "
        "Use wikipedia_tool for encyclopedic information. "
        "Use Gmail tools only when the user asks about their emails. "
        "You cannot send, delete, or modify emails."
    )
)


# --------------------------------------------------
# ASR EVALUATION
# --------------------------------------------------

wer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

cer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars()
])


def evaluate_asr(reference_text, hypothesis_text):
    try:
        wer_score = jiwer.wer(
            reference_text,
            hypothesis_text,
            reference_transform=wer_transform,
            hypothesis_transform=wer_transform
        )

        cer_score = jiwer.cer(
            reference_text,
            hypothesis_text,
            reference_transform=cer_transform,
            hypothesis_transform=cer_transform
        )

        print("\nASR Evaluation Metrics:")
        print(f"WER: {wer_score:.4f}")
        print(f"CER: {cer_score:.4f}")
        return wer_score, cer_score

    except Exception as e:
        print(f"ASR Evaluation Error: {e}")
        return None, None


# --------------------------------------------------
# MAIN VOICE ASSISTANT
# --------------------------------------------------

try:
    print("\nJJ is ready. Say 'Hey JJ' to wake me up.")

    while True:
        request_id = str(uuid.uuid4())
        request_start = time.perf_counter()

        try:
            log_event(request_id, "application", "started")

            print("\n-----------------------------------------")
            print("Listening..." if assistant_awake else "Waiting for Hey JJ...")

            # --------------------------------------
            # RECORD AUDIO
            # --------------------------------------

            audio_start = time.perf_counter()

            with sr.Microphone(sample_rate=16000) as source:
                audio = recognizer.listen(
                    source,
                    timeout=START_TIMEOUT,
                    phrase_time_limit=MAX_RECORDING
                )

            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())

            audio_latency = time.perf_counter() - audio_start
            log_event(
                request_id, "audio_recording", "success",
                latency_seconds=round(audio_latency, 3)
            )

            # --------------------------------------
            # WHISPER TRANSCRIPTION
            # --------------------------------------

            print("Processing your voice...")
            transcription_start = time.perf_counter()

            with open(filename, "rb") as audiofile:
                user_input = client.audio.transcriptions.create(
                    file=(filename, audiofile.read()),
                    model="whisper-large-v3-turbo",
                    language="en"
                )

            spoken_text = user_input.text.strip()
            original_text = spoken_text
            print(f"You Said: {spoken_text}")

            transcription_latency = time.perf_counter() - transcription_start
            log_event(
                request_id, "transcription", "success",
                latency_seconds=round(transcription_latency, 3)
            )

            if not spoken_text:
                print("I didn't catch that. Listening again...")
                continue

            # --------------------------------------
            # OPTIONAL ASR EVALUATION
            # --------------------------------------

            if DEBUG_MODE:
                reference_phrase = input(
                    "\nEnter the exact sentence you spoke (Enter to skip): "
                ).strip()

                if reference_phrase:
                    wer_score, cer_score = evaluate_asr(reference_phrase, original_text)
                    if wer_score is not None and cer_score is not None:
                        log_event(
                            request_id, "asr_evaluation", "success",
                            wer=wer_score, cer=cer_score
                        )
                    else:
                        log_event(request_id, "asr_evaluation", "failed")
                else:
                    log_event(request_id, "asr_evaluation", "skipped")

            # --------------------------------------
            # WAKE WORD DETECTION
            # --------------------------------------

            heard = re.sub(r"[^a-z0-9\s]", " ", spoken_text.lower())
            heard = " ".join(heard.split())

            if not assistant_awake:
                matched_wake_word = next(
                    (word for word in wake_words if word in heard),
                    None
                )

                if not matched_wake_word:
                    print("Wake word not detected. Still listening...")
                    log_event(request_id, "wake_word", "not_detected")
                    continue

                assistant_awake = True
                log_event(request_id, "wake_word", "detected")

                # Allow both 'Hey JJ' and 'Hey JJ, what is Python?'
                spoken_text = re.sub(
                    re.escape(matched_wake_word),
                    "",
                    heard,
                    count=1
                ).strip()

                if not spoken_text:
                    speak("Yeah, I'm here. What's up?")
                    continue

                speak("I'm listening.")

            # --------------------------------------
            # STANDBY / EXIT COMMANDS
            # --------------------------------------

            command = spoken_text.lower().strip(" .!?")

            if command in [" Hi jj", "Hello jay"]:
                speak("Alright, I'll be here when you need me.")
                assistant_awake = False
                log_event(request_id, "application", "standby")
                continue

            if command in ["goodbye jj", "goodbye jay jay", "go to sleep", "sleep jj","stop jj"]:
                speak("Okay, shutting down. See you later!")
                log_event(request_id, "application", "user_exit")
                break

            # -------------------------------------- 
            # LANGGRAPH AGENT
            # --------------------------------------

            print(f"Your Question: {spoken_text}")
            agent_start = time.perf_counter()

            response = agent.invoke(
                {"messages": [{"role": "user", "content": spoken_text}]},
                config={"configurable": {"thread_id": "voice_session_1"}}
            )

            assistant_answer = response["messages"][-1].content
            agent_latency = time.perf_counter() - agent_start

            log_event(
                request_id, "agent", "success",
                latency_seconds=round(agent_latency, 3)
            )

            # --------------------------------------
            # TEXT TO SPEECH
            # --------------------------------------

            tts_start = time.perf_counter()
            speak(assistant_answer)
            tts_latency = time.perf_counter() - tts_start

            log_event(
                request_id, "tts", "attempted",
                latency_seconds=round(tts_latency, 3)
            )

            # --------------------------------------
            # TOTAL LATENCY
            # --------------------------------------

            total_latency = time.perf_counter() - request_start
            log_event(
                request_id, "application", "completed",
                total_latency_seconds=round(total_latency, 3)
            )

        except sr.WaitTimeoutError:
            print("No speech detected. Listening again...")
            log_event(request_id, "audio_recording", "timeout")
            continue

        except Exception as e:
            log_event(
                request_id, "application", "failed",
                error_type=type(e).__name__
            )
            print(f"\nAn error occurred: {e}")
            print("Let's try again...")

except KeyboardInterrupt:
    print("\nAssistant stopped manually. Goodbye!")

finally:
    connection.close()
    print("PostgreSQL connection closed.")
