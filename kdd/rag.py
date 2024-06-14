import bz2
import json
import pandas as pd
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
import numpy as np
import ray
from collections import defaultdict

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids
    







def generate_answer(self, query: str, search_results: List[Dict]) -> str:
    """
    Generate an answer based on a provided query and a list of pre-cached search results.

    Parameters:
    - query (str): The user's question or query input.
    - search_results (List[Dict]): A list containing the search result objects,
    as described here:
        https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail

    Returns:
    - (str): A plain text response that answers the query. This response is limited to 75 tokens.
        If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

    Notes:
    - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
        the penalty for hallucination.
    - Response Time: Ensure that your model processes and responds to each query within 10 seconds.
        Failing to adhere to this time constraint **will** result in a timeout during evaluation.
    """
    # Default response when unsure about the answer
    answer = "i don't know"


    GOOGLE_API_KEY=os.env("GOOGLE_API_KEY") 

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

    # Initialize a list to hold all extracted sentences from the search results.
    all_sentences = []

    # Process each HTML text from the search results to extract text content.
    for html_text in search_results:
        # Parse the HTML content to extract text.
        soup = BeautifulSoup(
            html_text["page_result"], features="html.parser"
        )
        text = soup.get_text().replace("\n", "")
        if len(text) > 0:
            # Convert the text into sentences and extract their offsets.
            offsets = text_to_sentences_and_offsets(text)[1]
            for ofs in offsets:
                # Extract each sentence based on its offset and limit its length.
                sentence = text[ofs[0] : ofs[1]]
                all_sentences.append(
                    sentence[: self.max_ctx_sentence_length]
                )
        else:
            # If no text is extracted, add an empty string as a placeholder.
            all_sentences.append("")

    references = "\n".join(all_sentences)
    response = model.generate_content(self.prompt_template.format(query=query, references=references))
    print(response.text)
    answer = response.text
    time.sleep(20)
    # Trim prediction to a max of 75 tokens
    trimmed_answer = trim_predictions_to_max_token_length(answer)
