from openai import OpenAI

base_url = "https://api.aimlapi.com/v1"

# Create OpenAI client with AIML API base URL
client = OpenAI(
    api_key="46d3e6920b494f248a069cab79e4f817",
    base_url="https://api.aimlapi.com/v1"
)

try:
    # Test the connection
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a travel agent. Be descriptive and helpful"},
            {"role": "user", "content": "Tell me about San Francisco"}
        ]
    )
    print("API connection successful!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", str(e))

# Insert your AIML API key in the quotation marks instead of <YOUR_AIMLAPI_KEY>:
api_key = "46d3e6920b494f248a069cab79e4f817" 

system_prompt = "You are a travel agent. Be descriptive and helpful."
user_prompt = "Tell me about San Francisco"

api = OpenAI(api_key=api_key, base_url=base_url)


def main():
    completion = api.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    response = completion.choices[0].message.content

    print("User:", user_prompt)
    print("AI:", response)


if __name__ == "__main__":
    main()