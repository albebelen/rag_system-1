import os
import json
from tqdm import tqdm
from pageindex import PageIndexClient
import pageindex.utils as utils
from utils.model_factories import create_default_model


def _parse_tree_search_result(response):
    response = response.strip()
    if response.startswith("```"):
        response = response.removeprefix("```").removeprefix("json").strip()
        response = response.removesuffix("```").strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        if start == -1:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(response[start:])
        return parsed


def retrieve_dataset(doc_ids, dataset):
    if "PAGE_INDEX_API_KEY" not in os.environ:
        raise "missing PAGE_INDEX_API_KEY"
   
    pi_client = PageIndexClient(api_key=os.environ.get("PAGE_INDEX_API_KEY"))
    llm = create_default_model()

    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]

    trees = []
    for doc_id in doc_ids:
        if not pi_client.is_retrieval_ready(doc_id):
            raise "Document was not processed"
        trees.append(pi_client.get_tree(doc_id, node_summary=True)['result'])

    contexts = []
    for query in tqdm(dataset["question"], desc="Retrieving tree"):
        contexts.append([retrieve(tree, llm, query) for tree in trees])
        
    dataset["contexts"] = contexts
    dataset["retrieved_contexts"] = contexts
    return dataset

def retrieve(tree, llm, query):
    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all nodes that are likely to contain the answer to the question.

    Question: {query}

    Document tree structure:
    {json.dumps(tree_without_text, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant to the question>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    tree_search_result = llm.invoke(search_prompt).text
    try:
        result = _parse_tree_search_result(tree_search_result)
        node_list = result["node_list"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        preview = tree_search_result[:200].replace("\n", " ")
        raise ValueError(f"Invalid PageIndex retrieval response: {preview!r}") from exc

    node_map = utils.create_node_mapping(tree)
    return "\n\n".join(
        node_map[node_id]["text"] for node_id in node_list if node_id in node_map
    )
