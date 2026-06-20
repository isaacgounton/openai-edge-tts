# tts_handler.py

import edge_tts
import asyncio
import tempfile
import subprocess
import threading
import os
from pathlib import Path

from utils import DETAILED_ERROR_LOGGING
from config import DEFAULT_CONFIGS

# Language default (environment variable)
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', DEFAULT_CONFIGS["DEFAULT_LANGUAGE"])

# OpenAI voice names mapped to edge-tts equivalents
voice_mapping = {
    'alloy': 'en-US-JennyNeural',
    'ash': 'en-US-AndrewNeural',
    'ballad': 'en-GB-ThomasNeural',
    'coral': 'en-AU-NatashaNeural',
    'echo': 'en-US-GuyNeural',
    'fable': 'en-GB-SoniaNeural',
    'nova': 'en-US-AriaNeural',
    'onyx': 'en-US-EricNeural',
    'sage': 'en-US-JennyNeural',
    'shimmer': 'en-US-EmmaNeural',
    'verse': 'en-US-BrianNeural',
}

model_data = [
        {"id": "tts-1", "name": "Text-to-speech v1"},
        {"id": "tts-1-hd", "name": "Text-to-speech v1 HD"},
        {"id": "gpt-4o-mini-tts", "name": "GPT-4o mini TTS"}
    ]

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

async def _generate_audio_stream(text, voice, speed):
    """Generate streaming TTS audio using edge-tts."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is
    
    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"
    
    # Create the communicator for streaming
    communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
    
    # Stream the audio data
    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

def generate_speech_stream(text, voice, speed=1.0):
    """Generate streaming speech audio (synchronous generator wrapper).

    `_generate_audio_stream` is an async *generator*, so `asyncio.run()` on it
    raised "a coroutine was expected, got <async_generator>" and broke the SSE
    path (issue #34). Drive it from a dedicated event loop and yield chunks
    synchronously instead.
    """
    async_generator = _generate_audio_stream(text, voice, speed)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        while True:
            try:
                next_chunk = loop.run_until_complete(async_generator.__anext__())
            except StopAsyncIteration:
                break
            yield next_chunk
    finally:
        # Best-effort cleanup of async generators and loop
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


def generate_pcm_stream(text, voice, speed=1.0, sample_rate=24000):
    """Yield raw little-endian 16-bit mono PCM as edge-tts produces it.

    This is the low-latency path for real-time voice agents: edge-tts emits mp3
    chunks as Microsoft generates them; we pipe those straight into ffmpeg
    (mp3 -> s16le) and yield decoded PCM the moment it is available — no
    save-to-file, no full-utterance buffering, no separate convert pass. First
    audio lands in ~300ms instead of after the whole clip is synthesized.

    A background thread runs the edge-tts asyncio stream and feeds ffmpeg's
    stdin; the calling (request) thread reads ffmpeg's stdout and yields. The
    threaded WSGI server gives each request its own thread, so the blocking
    pipe reads here never stall other requests.
    """
    edge_tts_voice = voice_mapping.get(voice, voice)
    try:
        speed_rate = speed_to_rate(speed)
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "mp3", "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(sample_rate), "pipe:1",
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )

    def _feed() -> None:
        async def _run() -> None:
            communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
            async for chunk in communicator.stream():
                if chunk["type"] == "audio" and chunk.get("data"):
                    ffmpeg.stdin.write(chunk["data"])
        try:
            asyncio.run(_run())
        except Exception as e:  # never leave the reader hanging
            print(f"Error feeding edge-tts -> ffmpeg: {e}")
        finally:
            try:
                ffmpeg.stdin.close()
            except Exception:
                pass

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()

    try:
        while True:
            data = ffmpeg.stdout.read(4096)
            if not data:
                break
            yield data
    finally:
        try:
            ffmpeg.stdout.close()
        except Exception:
            pass
        feeder.join(timeout=2)
        if ffmpeg.poll() is None:
            ffmpeg.terminate()

async def _generate_audio(text, voice, response_format, speed):
    """Generate TTS audio and optionally convert to a different format."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is

    # Generate the TTS output in mp3 format first
    temp_mp3_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3_file_obj.name

    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"

    # Generate the MP3 file with explicit error handling + cleanup (borrowed
    # from the dev_june branch): on any edge-tts failure, remove the temp file
    # and surface a clear RuntimeError instead of leaking a half-written file.
    try:
        print(f"Generating TTS audio with voice: {edge_tts_voice}, rate: {speed_rate}")
        communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
        await communicator.save(temp_mp3_path)
        temp_mp3_file_obj.close()  # Explicitly close our file object for the initial mp3
        print(f"TTS audio generated successfully, file size: {os.path.getsize(temp_mp3_path)} bytes")
    except Exception as e:
        temp_mp3_file_obj.close()
        Path(temp_mp3_path).unlink(missing_ok=True)
        error_msg = f"Error generating TTS audio with voice '{edge_tts_voice}': {str(e)} (type: {type(e).__name__})"
        print(error_msg)
        raise RuntimeError(error_msg)

    # If the requested format is mp3, return the generated file directly
    if response_format == "mp3":
        return temp_mp3_path

    # Check if FFmpeg is installed
    if not is_ffmpeg_installed():
        print("FFmpeg is not available. Returning unmodified mp3 file.")
        return temp_mp3_path # Return the original mp3 path, it won't be cleaned by this function

    # Create a new temporary file for the converted output
    converted_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
    converted_path = converted_file_obj.name
    converted_file_obj.close() # Close file object, ffmpeg will write to the path

    # Build the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", temp_mp3_path,  # Input file path
        "-c:a", {
            "aac": "aac",
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "opus": "libopus",
            "flac": "flac",
            "pcm": "pcm_s16le",
        }.get(response_format, "aac"),  # Default to AAC if unknown
    ]

    if response_format not in ("wav", "pcm"):
        ffmpeg_command.extend(["-b:a", "192k"])

    ffmpeg_command.extend([
        "-f", {
            "aac": "mp4",  # AAC in MP4 container
            "mp3": "mp3",
            "wav": "wav",
            "opus": "ogg",
            "flac": "flac",
            "pcm": "s16le",  # raw 16-bit PCM
        }.get(response_format, response_format),  # Default to matching format
        "-y",  # Overwrite without prompt
        converted_path  # Output file path
    ])

    try:
        # Run FFmpeg command and ensure no errors occur
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Clean up potentially created (but incomplete) converted file
        Path(converted_path).unlink(missing_ok=True)
        # Clean up the original mp3 file as well, since conversion failed
        Path(temp_mp3_path).unlink(missing_ok=True)
        
        if DETAILED_ERROR_LOGGING:
            error_message = f"FFmpeg error during audio conversion. Command: '{' '.join(e.cmd)}'. Stderr: {e.stderr.decode('utf-8', 'ignore')}"
            print(error_message) # Log for server-side diagnosis
        else:
            error_message = f"FFmpeg error during audio conversion: {e}"
            print(error_message) # Log a simpler message
        raise RuntimeError(f"FFmpeg error during audio conversion: {e}") # The raised error will still have details via e

    # Clean up the original temporary file (original mp3) as it's now converted
    Path(temp_mp3_path).unlink(missing_ok=True)

    return converted_path

def generate_speech(text, voice, response_format, speed=1.0):
    return asyncio.run(_generate_audio(text, voice, response_format, speed))

def get_models():
    return model_data

def get_models_formatted():
    return [{ "id": x["id"], "object": "model", "owned_by": "openai-edge-tts" } for x in model_data]

def get_voices_formatted():
    return [{ "id": k, "name": v } for k, v in voice_mapping.items()]

async def _get_voices(language=None):
    """List all voices, filter by language if specified. Handle connection
    errors gracefully (borrowed from dev_june): if edge-tts's voice listing
    fails, return a small fallback set so callers never get an empty/500."""
    try:
        all_voices = await edge_tts.list_voices()
        language = language or DEFAULT_LANGUAGE  # Use default if no language specified
        filtered_voices = [
            {"name": v['ShortName'], "gender": v['Gender'], "language": v['Locale']}
            for v in all_voices if language == 'all' or language is None or v['Locale'] == language
        ]
        return filtered_voices
    except Exception as e:
        print(f"Error listing voices: {e}")
        fallback_voices = [
            {"name": "en-US-AvaNeural", "gender": "Female", "language": "en-US"},
            {"name": "en-US-AndrewNeural", "gender": "Male", "language": "en-US"},
            {"name": "en-GB-SoniaNeural", "gender": "Female", "language": "en-GB"},
            {"name": "fr-FR-RemyMultilingualNeural", "gender": "Male", "language": "fr-FR"},
            {"name": "fr-FR-VivienneMultilingualNeural", "gender": "Female", "language": "fr-FR"},
            {"name": "es-ES-ElviraNeural", "gender": "Female", "language": "es-ES"},
        ]
        if language == 'all' or language is None:
            return fallback_voices
        return [v for v in fallback_voices if v['language'] == language]

def get_voices(language=None):
    return asyncio.run(_get_voices(language))

def speed_to_rate(speed: float) -> str:
    """
    Converts a multiplicative speed value to the edge-tts "rate" format.
    
    Args:
        speed (float): The multiplicative speed value (e.g., 1.5 for +50%, 0.5 for -50%).
    
    Returns:
        str: The formatted "rate" string (e.g., "+50%" or "-50%").
    """
    if speed < 0 or speed > 2:
        raise ValueError("Speed must be between 0 and 2 (inclusive).")

    # Convert speed to percentage change
    percentage_change = (speed - 1) * 100

    # Format with a leading "+" or "-" as required
    return f"{percentage_change:+.0f}%"
