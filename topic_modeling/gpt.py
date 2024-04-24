import os
import openai
import time
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

client = openai.AzureOpenAI(
    api_version = "2023-07-01-preview",
    azure_endpoint = 'https://genai-nexus.api.corpinter.net/apikey/',
    api_key = os.getenv('NEXUS_API_KEY'),
)

batch_size = 25
pause_duration = 10

responses = []

df = pd.read_csv("topic_modeling/sectioned_data_2022.csv")

for index, row in df.iloc[:5].iterrows():
    sentence = row['overwegingen']
    ecli = row['ecli']
    prompt = "Kun je het volgende stuk tekst opsplitsen in segmenten gebaseerd op nummer? Dus 1, 2, 3, 4 etc. zijn verschillende stukjes, ook subkopjes zoals 1.1, 1.2 etc. horen bij 1.\n" + sentence + "\nZou je ieder segment vervolgens kunnen toewijzen aan een van de volgende juridische onderwerpen:\n feiten en omstandigheden, voorgaande juridische acties en besluiten (ook onderscheid tussen verweerder, appellante en de verschillende juridische lichamen zoals, rechtbank, gerechtshof etc.), standpunten appellante, standpunten verweerder, juridische middelen, beoordeling door het college.\n Schrijf het resultaat op als een een lijst dat begint met het nummer van het tekst segment. De opmaak van de lijst moet niet genummerd zijn of opsommingstekens hebben, zet tussen elk onderwerp 2 newlines en elk segment behorende tot een onderwerp 1 newline. Vermeld niks anders dan het onderwerp en de nummers van de segmenten (ook niet de tekst van de segmenten)."

    try:
        messages = [
            {
                "role" :"user",
                "content" :  prompt,
            },
        ]

        completion = client.chat.completions.create(
            model = "gpt4-turbo",
            messages = messages,
            temperature = 0,
            max_tokens = 4096,
            top_p = 0.1,
            frequency_penalty = 0,
            presence_penalty = 0,
            stop = None,
            logit_bias = {},
        )

        response_content = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing sentence: {sentence}")
        print(e)
        response_content = None
    
    responses.append((ecli, sentence, response_content))

    if response_content is not None:
        print(f"Sentence: {sentence}")
        print("Response: " + response_content)

    if (index + 1) % batch_size == 0:
        print(f"Processsed {index + 1} sentences taking a {pause_duration}-second break...")
        time.sleep(pause_duration)

responses_df = pd.DataFrame(responses, columns=['ecli','sentence', 'response'])
responses_df.to_csv("responses.csv", index=False)