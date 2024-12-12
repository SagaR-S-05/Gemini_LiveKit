import asyncio

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
import google.generativeai as genai  # Import Gemini API

load_dotenv()


async def entrypoint(ctx: JobContext):
    try:
        # Initialize the chat context for the assistant
        if hasattr(llm, "ChatContent") and callable(llm.ChatContent):
            # Safely initialize ChatContent
            initial_ctx = llm.ChatContent().append(
                role="system",
                text=(
                    "You are a voice assistant created by LiveKit using the Gemini API. "
                    "Your interface with users will be voice. You should use short and concise responses, "
                    "avoiding usage of unpronounceable punctuation."
                ),
            )
        else:
            raise ValueError("llm.ChatContent is not properly defined or callable.")

        # Connect to the context with audio subscription
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Configure Gemini API for LLM responses
        gemini_llm = genai.GenerativeLLM()  # Assuming generativeai has a class for LLM usage

        # Create the voice assistant using Gemini
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),  # Silero for Voice Activity Detection
            stt=genai.SpeechToText(),  # Assuming generativeai has STT capabilities
            llm=gemini_llm,  # Use Gemini for LLM responses
            tts=genai.TextToSpeech(),  # Assuming generativeai has TTS capabilities
            chat_ctx=initial_ctx,
        )
        assistant.start(ctx.room)

        # Greet the user
        await asyncio.sleep(1)
        await assistant.say("Hey, how can I help you today?", allow_interruptions=True)
    except Exception as e:
        # Log the error and provide a clear traceback
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
