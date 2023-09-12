import pickle
import difflib
from collections import Counter
from pprint import pprint

def keyword_search(text_list, keywords, n):
    """
    Perform keyword search against a list of strings and return the "n" best results.
    
    Parameters:
        text_list (list of str): The list of texts to search in.
        keywords (list of str): The keywords to search for.
        n (int): The number of best results to return.
        
    Returns:
        list of str: The "n" best-matching texts.
    """
    # Initialize a Counter object to store the text and its score
    scores = Counter()
    
    # Loop through each text in the text list
    for text in text_list:
        # Tokenize the text into words
        words = text.lower().split()
        
        # Initialize a score variable for this text
        score = 0
        
        # Loop through each keyword
        for keyword in keywords:
            # Count the occurrence of the keyword in the text
            score += words.count(keyword.lower())
            
        # Store the score for this text
        scores[text] = score
    
    # Get the "n" best-matching texts based on score
    best_matches = [text for text, _ in scores.most_common(n)]
    
    return best_matches

if __name__ == "__main__":
    with open("/home/awang/logbook_processed.pkl", 'rb') as f:
        data = pickle.load(f)
    
    res = keyword_search(data.values(), ["ICRF", "sputtering"], 20)
    pprint(res)
    import pdb; pdb.set_trace()