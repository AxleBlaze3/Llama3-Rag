import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

######################################################################################################
######################################################################################################
###
### IMPORTANT !!!
### Before submitting, please follow the instructions in the docs below to download and check in :
### the model weighs.
###
###  https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/download_baseline_model_weights.md
###
### And please pay special attention to the comments that start with "TUNE THIS VARIABLE"
###                        as they depend on your model and the available GPU resources.
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
# for more information on the MockAPI.
#
# **Note**: This environment variable will not be available for Task 1 evaluations.
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
#NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
#MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
#MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
#AICROWD_SUBMISSION_BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
#VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
#VLLM_GPU_MEMORY_UTILIZATION = 0.8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


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
VLLM_GPU_MEMORY_UTILIZATION = 0.8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32

#### CONFIG PARAMETERS END---

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

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "models/meta-llama/llama-3"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """
            )

        # Initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "models/sentence-transformers/mixedbread-large",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            #print(_idx, interaction_id)
            query = queries[_idx]
            #print(query)
            query_time = query_times[_idx]
            #print(query_time)
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            
            #print(retrieval_results)
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
            

            #print(batch_retrieval_results)
        # Prepare formatted prompts from the LLM     
            
            ### Getting Clues

        formatted_prompts = self.format_prompts(queries, query_times, message_no=0, batch_retrieval_results=batch_retrieval_results)
        responses = self.llm.generate(
              formatted_prompts,
              vllm.SamplingParams(
                  n=1,  # Number of output sequences to return for each prompt.
                  top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                  temperature=0.01,  # Randomness of the sampling
                  skip_special_tokens=True,  # Whether to skip special tokens in the output.
                  max_tokens=2096,  # Maximum number of tokens to generate per output sequence.
                  
                  # Note: We are using 50 max new tokens instead of 75,
                  # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
                  # Llama3 instead uses a different tokenizer with a larger vocabulary
                  # This allows the Llama3 tokenizer to represent the same content more efficiently, 
                  # while using fewer tokens.
              ),
              use_tqdm=False # you might consider setting this to True during local development
            )

        clues_list=[]
        for response in responses:
              
              clues_list.append(response.outputs[0].text)

            ### Asking for 3 detailed answers
        answerslist=[]
        response_list=[]

        for steps in range(0, 3):
          formatted_prompts = self.format_prompts(queries, query_times, message_no=1, context=clues_list)
          responses = self.llm.generate(
              formatted_prompts,
              vllm.SamplingParams(
                  n=1,  # Number of output sequences to return for each prompt.
                  top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                  temperature=0.5,  # Randomness of the sampling
                  skip_special_tokens=True,  # Whether to skip special tokens in the output.
                  max_tokens=3126,  # Maximum number of tokens to generate per output sequence.
                  
                  # Note: We are using 50 max new tokens instead of 75,
                  # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
                  # Llama3 instead uses a different tokenizer with a larger vocabulary
                  # This allows the Llama3 tokenizer to represent the same content more efficiently, 
                  # while using fewer tokens.
              ),
              use_tqdm=False  # you might consider setting this to True during local development
          )

          if steps==0:
            answerslist=[f'\nAnswer {steps+1}:\n'+response.outputs[0].text for response in responses]

          if steps!=0:
            response_list=[f'\nAnswer {steps+1}:\n'+response.outputs[0].text for response in responses]
            answerslist = [a + response for a, response in zip(answerslist, response_list)]

          
          #for response in responses:
           #   det_answer = response.outputs[0].text
            #  print(det_answer)
             # answerslist.append(det_answer)


          ### Asking for self consistency final answer
        #det_answers="\n\n".join(f"Answer {index + 1}: {element}" for index, element in enumerate(answerslist))
        formatted_prompts = self.format_prompts(queries, query_times, message_no=2, context=answerslist)

            # Generate responses via vllm

        responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.01,  # Randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=512,  # Maximum number of tokens to generate per output sequence.
                    
                    # Note: We are using 50 max new tokens instead of 75,
                    # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
                    # Llama3 instead uses a different tokenizer with a larger vocabulary
                    # This allows the Llama3 tokenizer to represent the same content more efficiently, 
                    # while using fewer tokens.
                ),
                use_tqdm=False # you might consider setting this to True during local development
            )

            #finalanswer=response.outputs[0].text

        # Aggregate answers into List[str]
        answers = []
        for response in responses:
            answers.append(response.outputs[0].text)
            
        return answers

    def format_prompts(self, queries, query_times, message_no, batch_retrieval_results=[], context=None):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        

        messages_clues="""You will be given a context document and a question from the user. You must extract the most relevant information as clues to the user.

        The clues should be any well-reasoned and identified piece of information that either directly answers the question or provides information that could be used by the user to get the answer on his own.
        If math is required, such as finding averages, percentages, aggregation of data, you will compute it and present the result as the clue as well.

        You will not add anything of your knowledge to the clues EVER. Neither will you ever make use of any other source or context aside from what the user provides. Stick to only the information in the context.

        Return the well-reasoned and detailed clues gathered from the provided context document regarding the query. 
        You must be concise and precise, do not go into long explanation if you dont have the answer. Only explain when the answer is present in the context."""

        messages_answer_majority="""You are an expert at finding the most relevant pieces of texts for anything that a user may ask.
      You will be provided a question that the user wants answered and also the clues to answer the question.

      Your job is to read all clues and pick the majority opinion from them as the answer to the question that the user requires.
      If there are contradictory answers, you will still pick the majority opinion from the clues as the final answer.
      The answer should be based on the majority opinion from the clues provided, DO NOT ADD YOUR OWN INFORMATION NEITHER USE EXTERNAL SOURCES.

      You dont need to be super critical of the clues. You may consider it reliable source of information.
      You will scour the clues and find the answer. Sometimes you will need to collect information from different clues to piece the answer.

      You will return the majority answer in detail at the very start. You will end with:
      ###Final Answer: <your answer>

      If you are absolutely sure, without a shadow of a doubt, that the answer does not exist in any of the contexts, then reply with only... 'I dont know'. """

        messages_self_con="""You are a judge that favors the majority. You will be given a question and 3 detailed answers for the question. 
      Your job is to look at the "Final Answer" section of the 3 answers and pick the common answer amongst the majority as the true final answer.
      Only provide this final answer in under 50 tokens that actually properly answers the question, nothing else, just one small sentence.

      If there is no conclusive final answer in the detailed answer, then you will simply return "I don't know." as well. You do not need to explain why you chose an answer."""

        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]

            user_message = ""
            references = ""
            
            # Limit the length of references to fit the model's input size.

            #user_message += f"{references}\n------\n\n"
            #user_message 
            #user_message += f"Using only the references listed above, answer the following question: \n"
            #user_message += f"Current Time: {query_time}\n"
            #user_message += f"Question: {query}\n"

            #user_message += f"{references}\n------\n\n"
            #user_message 
            #user_message += f"Using only the references listed above, answer the following question: \n"
            #user_message += f"Current Time: {query_time}\n"
            #user_message += f"Question: {query}\n"

            if message_no==0:
              retrieval_results = batch_retrieval_results[_idx]
              if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
              references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

              formatted_prompts.append(
                  self.tokenizer.apply_chat_template(
                      [
                          {"role": "system", "content": messages_clues},
                          {"role":"user", "content":f"""<References>{references}<\References>"""},
                          {"role":"user", "content":f"""Using the above references, provide clues for the following question: 
                          <User Question>{query}<User Question>
                          You will first repeat the question and the references, and only then provide the clues for it..."""}
                      ],
                      tokenize=False,
                      add_generation_prompt=True,
                  )
              )
            
            if message_no==1:
              clues=context[_idx]
              formatted_prompts.append(
                  self.tokenizer.apply_chat_template(
                      [
                          {"role": "system", "content": messages_answer_majority},
                          {"role":"user", "content":f"""<Clues>{clues}<\Clues>"""},
                          {"role":"user", "content":f""" 
                          <User Question>{query}<User Question>
                          You will first repeat the question, then the references and the clues, then give a detailed explanation for your answer..."""}
                      ],
                      tokenize=False,
                      add_generation_prompt=True,
                  )
              )

            if message_no==2:
              det_answers=context[_idx]
              formatted_prompts.append(
                  self.tokenizer.apply_chat_template(
                      [
                          {"role": "system", "content": messages_self_con},
                          {"role":"user", "content":f"""<Detailed Answers>{det_answers}<\Detailed Answers>"""},
                          {"role":"user", "content":f""" <User Question>{query}<User Question>"""}
                      ],
                      tokenize=False,
                      add_generation_prompt=True,
                  )
              )

        return formatted_prompts
