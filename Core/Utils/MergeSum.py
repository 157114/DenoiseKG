import asyncio
import tiktoken
from openai import AsyncOpenAI
from collections import Counter
from typing import Dict, Optional



GRAPH_FIELD_SEP = "<SEP>"
CONFIG_PATH = "./Option/Config2.yaml"

class GraphPrompt:
    SUMMARIZE_ENTITY_DESCRIPTIONS = """
You are a helpful assistant. Please summarize the following list of descriptions for the entity '{entity_name}' into a single, coherent paragraph.
Combine the key information and remove redundant details.

Descriptions to summarize:
{description_list}

Concise Summary:
"""
    SUMMARIZE_RELATION_DESCRIPTION = """
You are a helpful assistant. Please summarize the following list of descriptions for the relationship '{item_name}' into a single, coherent paragraph.
Combine the key information and remove redundant details.

Descriptions to summarize:
{description_list}

Concise Summary:
"""
    SUMMARIZE_KEYWORDS = """
You are a helpful assistant. The following is a long, repetitive list of keywords for the relationship '{item_name}'.
Your task is to de-duplicate the list and distill it into a concise, representative set of the most important keywords, joined by "{separator}".

Keywords to summarize:
{keyword_list}

Concise and De-duplicated Keywords:
"""

class SummarizerConfig:
    def __init__(self, model_name: str):
        self.summary_max_tokens = 2000
        self.llm_model_max_token_size = 32678
        self.summarization_model = model_name
        self.token_check_threshold = 2000



class DescriptionSummarizer:
    def __init__(self, api_key: str, base_url: str, model_name: str, provider: str = "openai"):
        if not api_key:
            raise ValueError("OpenAI API key is required for summarization.")
        if provider == "dashscope" or model_name.startswith("qwen"):
            # Qwen via DashScope
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.config = SummarizerConfig(model_name)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def summarize(self, item_name: str, text_to_summarize: str, text_type: str) -> str:
        tokens = self.encoder.encode(text_to_summarize)
        if len(tokens) < self.config.token_check_threshold:
            return text_to_summarize

        if len(tokens) > self.config.llm_model_max_token_size:
            use_text = self.encoder.decode(tokens[:self.config.llm_model_max_token_size])
        else:
            use_text = text_to_summarize
        
        if text_type == 'entity_description':
            prompt_template = GraphPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS
            context_base = {"entity_name": item_name, "description_list": "\n".join(f"- {d.strip()}" for d in use_text.split(GRAPH_FIELD_SEP) if d.strip())}
        elif text_type == 'relation_description':
            prompt_template = GraphPrompt.SUMMARIZE_RELATION_DESCRIPTION
            context_base = {"item_name": item_name, "description_list": "\n".join(f"- {d.strip()}" for d in use_text.split(GRAPH_FIELD_SEP) if d.strip())}
        elif text_type == 'keywords':
            prompt_template = GraphPrompt.SUMMARIZE_KEYWORDS
            context_base = {"item_name": item_name, "keyword_list": ", ".join(sorted(set(k.strip() for k in use_text.split(GRAPH_FIELD_SEP) if k.strip()))), "separator": GRAPH_FIELD_SEP}
        else:
            print(f"   ⚠️ WARNING: Unknown text_type '{text_type}'. Skipping summarization.")
            return text_to_summarize

        prompt = prompt_template.format(**context_base)
        
        attempt=3
        while attempt>0:
            try:
                print(f" -> Triggering summarization for '{item_name}' ({text_type})...")
                response = await self.client.chat.completions.create(model=self.config.summarization_model, messages=[{"role": "user", "content": prompt}], max_tokens=self.config.summary_max_tokens, temperature=0.2)
                summary = response.choices[0].message.content
                return summary.strip() if summary else text_to_summarize
            except Exception as e:
                print(f"Attempt {attempt} failed: item_name '{item_name}': {e}.")
                attempt -= 1
        print(f"All attempts failed: item_name '{item_name}' ({text_type}). Returning original text.")
        return text_to_summarize
    
# --- MergeEntity and MergeRelationship classes remain the same as the previous "two-pass" version ---
# --- They are designed to just concatenate, which is what we want for Pass 1. ---
class MergeEntity:
    # ... (no changes from previous correct version)
    merge_keys = ["source_id", "entity_type", "description"]
    @staticmethod
    def merge_source_ids(existing_source_ids: str, new_source_ids: str): #...
        existing_list = existing_source_ids.split(GRAPH_FIELD_SEP) if existing_source_ids else []
        new_list = new_source_ids.split(GRAPH_FIELD_SEP) if new_source_ids else []
        merged_source_ids = list(set(new_list) | set(existing_list))
        return GRAPH_FIELD_SEP.join(sorted(merged_source_ids))
    @staticmethod
    def merge_types(existing_entity_types: str, new_entity_types: str): #...
        existing_list = existing_entity_types.split(GRAPH_FIELD_SEP) if existing_entity_types else []
        new_list = new_entity_types.split(GRAPH_FIELD_SEP) if new_entity_types else []
        merged_entity_types = existing_list + new_list
        entity_type_counts = Counter(merged_entity_types)
        most_common_type = entity_type_counts.most_common(1)[0][0] if entity_type_counts else ''
        return most_common_type
    @staticmethod
    def merge_descriptions(existing_descriptions: str, new_descriptions: str, summarizer: Optional[DescriptionSummarizer], entity_name: str) -> str: #...
        existing_list = existing_descriptions.split(GRAPH_FIELD_SEP) if existing_descriptions else []
        new_list = new_descriptions.split(GRAPH_FIELD_SEP) if new_descriptions else []
        merged_descriptions = list(set(new_list) | set(existing_list))
        description = GRAPH_FIELD_SEP.join(sorted(merged_descriptions))
        if summarizer:
            tokens = summarizer.encoder.encode(description)
            if len(tokens) > summarizer.config.token_check_threshold:
                return asyncio.run(summarizer.summarize(entity_name, description, text_type='entity_description'))
        return description
    @classmethod
    def merge_info(cls, existing_node_data, new_node_data, summarizer: Optional[DescriptionSummarizer] = None, entity_name: str = "Unknown"): #...
        merge_function_map = {"source_id": cls.merge_source_ids, "entity_type": cls.merge_types}
        merged_data = existing_node_data.copy()
        for key in cls.merge_keys:
            if key in existing_node_data and key in new_node_data:
                val1, val2 = existing_node_data.get(key), new_node_data.get(key)
                if key == "description":
                    merged_data[key] = cls.merge_descriptions(val1, val2, summarizer, entity_name)
                elif key in merge_function_map:
                    merged_data[key] = merge_function_map[key](val1, val2)
        return merged_data

class MergeRelationship:
    # ... (no changes from previous correct version)
    merge_keys = ["source_id", "weight", "description", "keywords", "relation_name"]
    merge_function = None
    @staticmethod
    def merge_weight(existing_weight, new_weight): #...
        return float(existing_weight or 0.0) + float(new_weight or 0.0)
    @staticmethod
    def merge_generic_field(existing_values: str, new_values: str): #...
        existing_list = existing_values.split(GRAPH_FIELD_SEP) if existing_values else []
        new_list = new_values.split(GRAPH_FIELD_SEP) if new_values else []
        return GRAPH_FIELD_SEP.join(sorted(set(existing_list + new_list)))
    @classmethod
    def merge_info(cls, existing_edge_data, new_edge_data): #...
        if cls.merge_function is None:
            cls.merge_function = {"weight": cls.merge_weight, "description": cls.merge_generic_field, "source_id": cls.merge_generic_field, "keywords": cls.merge_generic_field, "relation_name": cls.merge_generic_field}
        merged_data = existing_edge_data.copy()
        for key in cls.merge_keys:
            if key in existing_edge_data and key in new_edge_data:
                val1, val2 = existing_edge_data.get(key), new_edge_data.get(key)
                if val1 is not None and val2 is not None:
                    merged_data[key] = cls.merge_function[key](val1, val2)
        return merged_data