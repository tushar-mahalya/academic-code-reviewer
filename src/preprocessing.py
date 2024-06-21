import re


def cleanTranscript(text):
    # Replace non-breaking spaces and newlines
    cleaned_text = re.sub(r'\\xa0\\n|\\xa0\\xa0|\\n\\xa0|\\n', ' ', text)
    # Remove any extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
