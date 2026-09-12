import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils.my_log import logger

class DocumentSearchResponse(BaseModel):
    thinking: str = Field(description="Your thinking process on which nodes are relevant to the question")
    node_list: list[str] = Field(description="List of relevant node IDs found in the tree")

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are given a question and a tree structure of a document. "
        "Each node contains a node id, node title, and a corresponding summary. "
        "Your task is to find all nodes that are likely to contain the answer to the question."
        "Keep your 'thinking' explanation brief and concise (under 3 sentences)."
    ),
    (
        "human", 
        "Question: {query}\n\n"
        "Document tree structure:\n{tree_json}"
    )
])

def retrieve_dataset(doc_id, dataset):
    from pageindex import PageIndexClient
    from utils.model_factories import create_default_model
    from tqdm import tqdm

    if "PAGE_INDEX_API_KEY" not in os.environ:
        raise "missing PAGE_INDEX_API_KEY"
   
    pi_client = PageIndexClient(api_key=os.environ.get("PAGE_INDEX_API_KEY"))
    llm = create_default_model(max_tokens=2048)

    if not pi_client.is_retrieval_ready(doc_id):
        raise "Document was not processed"
    
    tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    contexts = []
    for query in tqdm(dataset["question"], desc="Retrieving tree"):
        contexts.append([ retrieve(tree, llm, query) ])
        
    dataset["contexts"] = contexts
    dataset["retrieved_contexts"] = contexts
    return dataset

def retrieve(tree, llm, query):
    import json
    import logging
    import pageindex.utils as utils
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])
    tree_json_str = json.dumps(tree_without_text, indent=2)
    node_map = utils.create_node_mapping(tree)

    structured_llm = llm.with_structured_output(DocumentSearchResponse)

    messages = prompt_template.invoke({
        "query": query, 
        "tree_json": tree_json_str
    }).to_messages()

    max_attempts = 3
    for attempt in range(max_attempts):
        is_last_attempt = attempt == max_attempts - 1

        try:
            response: DocumentSearchResponse = structured_llm.invoke(messages)
        except (ValidationError, OutputParserException) as e:
            if is_last_attempt:
                logging.error(f"Schema validation failed after {max_attempts} attempts: {e}")
                raise e
            # Extract the raw output if OutputParserException captured it
            raw_content = getattr(e, "llm_output", None) or getattr(e, "observation", None)
            messages.append(AIMessage(content=str(raw_content) if raw_content else "[Invalid JSON / Schema Output]"))
            messages.append(HumanMessage(
                content=f"Your previous response failed JSON/schema validation with this error:\n{e}\n"
                        f"Please correct your response to strictly match the requested JSON schema."
            ))
            continue

        resolved_texts = []
        invalid_ids = []
            
        for node_id in response.node_list:
            if node_id in node_map:
                resolved_texts.append(node_map[node_id]["text"])
            else:
                invalid_ids.append(node_id)
        
        if invalid_ids and not is_last_attempt:
            messages.append(AIMessage(content=response.model_dump_json(indent=2)))
            messages.append(HumanMessage(
                content=f"The node IDs {invalid_ids} do not exist in the document tree. "
                        f"Please review the tree structure and return only valid node IDs."
            ))
            continue
            
        if invalid_ids:
            logging.warning(f"Model returned some invalid node IDs {invalid_ids}, which could not be found in the tree node map. Ignoring.")
                
        return "\n\n".join(resolved_texts)
