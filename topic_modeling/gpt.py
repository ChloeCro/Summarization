import os
import openai
import time
import pandas as pd

from dotenv import load_dotenv

client = openai.AzureOpenAI(
    api_version = "2023-07-01-preview",
    azure_endpoint = 'https://genai-nexus.api.corpinter.net/apikey/',
    api_key = os.getenv('NEXUS_API_KEY'),
)

batch_size = 25
pause_duration = 10

responses = []

df = pd.read_csv("sectioned_data_2022.csv")

for index, row in df.iterrows():
    sentence = row['sentence']

    try:
        messages = [
            {
                "role" :"user",
                "content" : f"",
            },
        ]

        completion = client.chat.completions.create(
            model = "gpt4-turbo",
            messages = messages,
            temperature = 0,
            max_tokens = 50,
            top_p = 0.1,
            frequency_penalty = 0,
            presence_penalty = 0,
            stop = None,
            logit_bias = {},
        )

        responses_content = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing sentence: {sentence}")
        print(e)
        response_content = None
    
    responses.append((sentence, response_content))

    if response_content is not None:
        print(f"Sentence: {sentence}")
        print("Response: " + response_content)

    if (index + 1) % batch_size == 0:
        print(f"Processsed {index + 1} sentencesm taking a {pause_duration}-second break...")
        time.sleep(pause_duration)

responses_df = pd.DataFrame(responses, columns=['sentence', 'response'])
responses_df.to_csv("responses.csv", index=False)