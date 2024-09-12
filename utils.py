import textwrap

import fitz


def read_and_chunk_files(file_paths, chunk_size):

    all_chunks = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        words = text.split()
        chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]

        all_chunks.extend(chunks)

    return all_chunks


def extract_text_from_pdf(file_path):
    # Open the PDF document using the file path provided
    doc = fitz.open(file_path)
    # Initialize an empty string to store the extracted text
    text = ""
    # Loop through each page in the PDF document
    for page in doc:
        # Extract the text from the current page and append it to the 'text' string
        text += page.get_text()
    return text


def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))
