import pandas as pd
from collections import defaultdict
import ast
import re


df = pd.read_csv("responses.csv")
data = pd.read_csv("topic_modeling\sorted_sections.csv")

print(df)
print(data[:5])
data['organized'] = data['organized'].apply(ast.literal_eval)
organized_sections = data['organized'].tolist()
print(type(organized_sections[0]))
print(organized_sections[0])
"""
filtered_lists = []
pattern = r"^\s*\d"
for value in data['sections']:
    value_list = ast.literal_eval(value)

    # Filter elements that start with a number or a whitespace followed by a number
    filtered_elements = [element for element in value_list if re.match(pattern, element)]

    filtered_lists.append(filtered_elements)


print(filtered_lists)
"""
text_list = df['response'].tolist()
print(text_list)
ordered = []
for text in text_list:
    # Step 1: Split the text into individual lines
    lines = text.split("\n")

    # Step 2: Create a dictionary to hold the topics and their associated section numbers
    topics_dict = defaultdict(list)

    current_topic = None

    # Step 3: Loop through the lines and determine if it's a topic or a section number
    for line in lines:
        cleaned_line = line.strip()  # Remove extra spaces

        if re.match(r"^\d", cleaned_line):  # Check if the line starts with a digit
            if current_topic:
                topics_dict[current_topic].append(cleaned_line)  # Add the section to the current topic
        else:
            # If it's not a section number, it's likely a new topic
            current_topic = cleaned_line  # Set the current topic
    ordered.append(topics_dict)

# Output the dictionary
print(ordered)
print(len(ordered))

# Step 1: Create a dictionary to hold the joined section texts for each topic
topic_texts = {}

# Step 2: Loop over the topics dictionary and gather corresponding section texts
for topic, section_numbers in ordered[0].items():
    # Retrieve the texts for the given section numbers
    topic_sections = [organized_sections[0][int(section.split('.')[0])] for section in section_numbers if int(section.split('.')[0]) in organized_sections[0]]

    # Step 3: Join the section texts for each topic
    joined_text = ' '.join(topic_sections)  # Combine the texts into a single string
    topic_texts[topic] = joined_text  # Add to the dictionary

# Output the result
print(topic_texts)

# Step 1: Create a DataFrame from the topic_texts dictionary
joined = pd.DataFrame([topic_texts])

# Step 2: Save the DataFrame to a CSV file
joined.to_csv('topic_texts.csv', index=False)  # Don't include the index in the CSV file

# Output the path to the CSV file (if saving in a different environment)
print("CSV file saved as 'topic_texts.csv'")