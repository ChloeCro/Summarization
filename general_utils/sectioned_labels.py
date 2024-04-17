import pandas as pd
import ast
from collections import defaultdict

def organize_by_number(strings):
    result = defaultdict(list)
    for string in strings:
        period_index = string.find('.')
        if period_index != -1:
            number = string[:period_index]
            if number.isdigit():
                result[int(number)].append(string)

    # Join the lists into a single string per key
    for key in result:
        result[key] = ' '.join(result[key])

    # Sort the dictionary by keys to ensure order
    sorted_result = dict(sorted(result.items()))
    return sorted_result

def apply_to_dataframe(df, column_name):
    # Convert string representations of lists to actual lists
    df[column_name] = df[column_name].apply(ast.literal_eval)
    # Apply the organize_by_number function to each row's specified column
    df['organized'] = df[column_name].apply(organize_by_number)
    return df

# Reading the CSV file
data = pd.read_csv("sectioned_data_2022_test.csv")

# Applying the function to the DataFrame
organized_df = apply_to_dataframe(data, 'sections')
print(organized_df.organized[0])

# Saving the processed DataFrame to a new CSV file
organized_df.to_csv("sorted_sections.csv")
