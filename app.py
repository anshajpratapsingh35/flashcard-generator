import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL_1 = "openai/gpt-oss-120b"
MODEL_2 = "google/gemma-2-2b-it"
MODEL_ID = MODEL_1

SYSTEM_PROMPT = """You are a helpful study assistant.
Create exactly 5 study flashcards.
Each flashcard must contain exactly one question and one short answer.
Keep answers concise and suitable for a second-year computer science student.
Do not add introductions, conclusions, or unnecessary explanations.
Use exactly this format:

Flashcard 1
Question: ...
Answer: ...

Flashcard 2
Question: ...
Answer: ...

Flashcard 3
Question: ...
Answer: ...

Flashcard 4
Question: ...
Answer: ...

Flashcard 5
Question: ...
Answer: ...
"""

def get_token():
    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token == "YOUR_HF_TOKEN_HERE":
        raise RuntimeError("HF_TOKEN is missing. Open .env and add your Hugging Face token.")
    if not token.startswith("hf_"):
        raise RuntimeError("HF_TOKEN does not look like a valid Hugging Face token.")
    return token

def extract_text(response):
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and choice.message:
            content = getattr(choice.message, "content", "")
            if isinstance(content, str):
                return content.strip()
    return ""

def main():
    print("=" * 60)
    print("HUGGING FACE FLASHCARD GENERATOR")
    print("=" * 60)
    print(f"Model 1: {MODEL_1}")
    print(f"Model 2: {MODEL_2}")
    print(f"Current model: {MODEL_ID}")

    try:
        token = get_token()
        print(f"HF_TOKEN detected: {len(token)} characters")
    except Exception as e:
        print(f"\nCONFIGURATION ERROR: {e}")
        return

    topic = input("\nEnter a topic: ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    

    print("\nConnecting to Hugging Face...")
    print("Waiting for model response...\n")

    start = time.perf_counter()

    try:
        client = InferenceClient(
            api_key=token,
            provider="auto",
            timeout=60
        )

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}
            ],
            temperature=0.7,
            max_tokens=500
        )

        result = extract_text(response)

        if not result:
            print("ERROR: The model returned an empty response.")
            return

        latency = time.perf_counter() - start

        print("=" * 60)
        print("GENERATED FLASHCARDS")
        print("=" * 60)
        print(result)
        print("\n" + "=" * 60)
        print(f"Response time: {latency:.2f} seconds")
        print(f"Model: {MODEL_ID}")
        print("=" * 60)

    except Exception as e:
        latency = time.perf_counter() - start
        print("=" * 60)
        print("HUGGING FACE REQUEST FAILED")
        print("=" * 60)
        print(f"Time waited: {latency:.2f} seconds")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        print("=" * 60)

if __name__ == "__main__":
    main()
