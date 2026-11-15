import json
import requests
import sys
import uuid


API_URL = "http://localhost:8000/stream"

username = input("User Name: ")
session_id = str(uuid.uuid4())

print(f"Welcome {username}!")
print(f"Session ID: {session_id}")
print("Type 'exit' or 'quit' to leave.")

while True:
    question = input("\nAsk your question: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not question.strip():
        continue

    print("\nAssistant: ", end="", flush=True)

    try:
        with requests.post(
            API_URL,
            json={
                "username": username,
                "session_id": session_id,
                "message": question,
            },
            stream=True,
            timeout=(10, 300),
        ) as response:
            response.raise_for_status()

            current_event = None
            stream_finished = False

            for line in response.iter_lines(decode_unicode=True):

                if not line:
                    current_event = None
                    continue

                if line.startswith("event:"):
                    current_event = line[6:].strip()

                elif line.startswith("data:"):
                    payload = json.loads(line[5:].strip())

                    if current_event == "token":
                        content = payload.get("content", "")
                        print(content, end="", flush=True)

                    elif current_event == "done":
                        session_id = payload.get(
                            "session_id", session_id
                        )
                        stream_finished = True

                    elif current_event == "error":
                        print(
                            "\nStreaming error:",
                            payload.get("error", "Unknown error"),
                        )

            print()

            if stream_finished:
                print("Response completed.")


    except requests.exceptions.Timeout:
        print("\nRequest timed out.")

    except requests.exceptions.RequestException as exc:
        print(f"\nConnection error: {exc}")

    except json.JSONDecodeError:
        print("\nReceived invalid JSON from the streaming endpoint.")
