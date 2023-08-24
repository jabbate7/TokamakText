import re
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()

model = os.getenv("LLM_NAME")

# gpt-3.5-turbo has 4096 token length (including your full prompt + its response)
# rule of thumb from internet: token is about 0.75 of a word, or 4 characters
document_info={
    'shot': {'token_length': 300,
             'n_documents': 14},
    'run': {'token_length': 400,
            'n_documents': 10},
    'miniproposal': {'token_length': 400,
                     'n_documents': 10}
}
# overlap so we don't miss context due to splitting string at awkward spot
overlap_num_tokens=60

def make_document_dic_from_string(input_string, in_key, document_type):
    # split on any whitespace and remove stupidly repeated characters
    # if at least 3 symbols next to eachother, replace, with the first two
    # people often do =======, -----, etc. to separate stuff
    # people also do = = = or -- -- -- etc., should probably add this (slash find someone else's implementation)
    formatted_string=input_string
    formatted_string=re.sub(r'([^a-zA-Z0-9\s]{2})[^a-zA-Z0-9\s]+', '\1', formatted_string)
    # deals with '30,40,50,20,...'
    formatted_string=re.sub(r',', ', ', formatted_string)
    array_of_words=re.split(r'\s+', formatted_string)
    num_tokens=document_info[document_type]['token_length']
    string_of_documents=' '.join(array_of_words)
    encoder=tiktoken.encoding_for_model(model)
    encoded_documents=encoder.encode(string_of_documents)
    # the ".decode()" is just to convert bytes to strings, had to throw ignore in there to ignore bad characters (still haven't looked at which)
    document_array_of_decoded_tokens=[encoder.decode_single_token_bytes(token).decode('utf-8', errors='ignore') for token in encoded_documents]
    # right now it can split up the first and last word, maybe should enforce keeping these words intact
    array_of_documents=[''.join(document_array_of_decoded_tokens[i:i+num_tokens]) for i in range(0,
                                                                                                len(document_array_of_decoded_tokens)-num_tokens+1,
                                                                                                num_tokens - overlap_num_tokens)]
    if len(array_of_documents)==1:
        return {f'{document_type} {in_key}': array_of_documents[0]}
    else:
        return {f'{document_type} {in_key}_{i}': array_of_documents[i] for i in range(len(array_of_documents))}

