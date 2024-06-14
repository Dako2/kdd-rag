import bz2
import json
import pandas as pd

#from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from readability import Document

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available
max_ctx_sentence_length = 7000

#read jsonl file and convert it to dataframe
def read_jsonl_to_dataframe(file_path):
    data = []
    # Open the .jsonl file
    with open(file_path, 'r') as file:
        # Process each line
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

def read_bz2_to_dataframe(file_path):
    data = []
    # Open the .bz2 file
    with bz2.open(file_path, 'rt') as file:
        # Process each line
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

def possible_main_context_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Define the keywords to search for in class names
    keywords = ['content', 'main', 'post', 'paragraph', 'body', 'article']

    # Find all <p> tags with a class that matches one of the keywords
    p_tags_with_keyword_classes = soup.find_all('p', class_=lambda x: x and any(keyword in x for keyword in keywords))

    # Print the <p> tags and their content
    for p in p_tags_with_keyword_classes:
        print(p)
        print(f"Class(es): {p.get('class')}")
        print(f"Text: {p.get_text()}")
        print("\n---\n")

    # If you want to return the result as a list of strings (just the text content), you can do:
    text_contents = [p.get_text() for p in p_tags_with_keyword_classes]
    print(text_contents)
    return text_contents

# Path to your .bz2 file
file_path = 'dev_data.jsonl (2).bz2'
df = read_bz2_to_dataframe(file_path)
#file_path = 'crag_task_1_dev_v3_release.jsonl'
#df = read_jsonl_to_dataframe(file_path)
print(df.head())  # This prints the first few rows of the DataFrame

for x in range(len(df)):
    question = df['query'].iloc[x]
    print('\nQuestion:\n')
    print(question)
    print('\n\n\n\n')

    search_results = df['search_results'].iloc[x]
    for i in range(5):
        html_content = search_results[i]['page_result']
        
        if html_content:
            doc = Document(html_content)
            print(doc.title())
            main_content = doc.summary()        
            soup = BeautifulSoup(main_content, 'html.parser')
            final_text = soup.get_text().replace("\n\n\n\n", " ").strip()
            print(final_text)
 
        print("================ metadata ================")
        keys_to_save = ['page_name', 'page_url', 'page_snippet', 'page_last_modified']
        metadata = {key: search_results[i][key] for key in keys_to_save}
        metadata['page_id'] = i
        print(metadata)
        print("============== metadata end %s =============="%question)
        

