from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Get text response

response = client.responses.create(
    model="qwen3.8:latest",
    input="Tell me a joke that places Buffy Summers of the telivision show Buffy the Vampire Slayer in Quarks, the Ferengi bar on Deep Space Nine during the Cardasian occupation when Wesley Snipes dressed as Blade breaks through the door Katana in hand looking haunted and out of place.",
)

print(response.output_text)
