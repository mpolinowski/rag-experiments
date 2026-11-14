import requests
import uuid

API_URL = "http://localhost:8000/invoke"

username = input("User Name: ")
session_id = str(uuid.uuid4())

print(f"Welcome {username}!")
print(f"Session ID: {session_id}")

while True:
    question = input("\nAsk your question: ")

    if question.lower() in ["exit", "quit"]:
        break

    response = requests.post(
        API_URL,
        json={
            "username": username,
            "session_id": session_id,
            "message": question,
        },
        timeout=120,
    )

    response.raise_for_status()
    result = response.json()

    print("\nAssistant:", result["answer"])

    usage = result["usage"]
    print(
        f"Token usage: {usage['input_tokens']}, "
        f"{usage['output_tokens']}, "
        f"{usage['total_tokens']}"
    )
