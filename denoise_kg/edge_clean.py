import os
import re
import json, json_repair
import glob
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import time
import xml.etree.ElementTree as ET
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import argparse
import yaml
load_dotenv() 

MODEL_NAME = None
PRINTED_MODEL = False

def load_graphml_edges(input_file):
    """加载GraphML文件中的边信息"""
    try:
        
        G = nx.read_graphml(input_file)
        edges_data = []
        
        print(f"成功加载图数据，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        
        for source, target, data in G.edges(data=True):
            edge_info = {
                'source': str(source),
                'destination': str(target),
                'relationship': data.get('description')  
            }
            edges_data.append(edge_info)
            
        return edges_data
    except Exception as e:
        print(f"Error loading GraphML file: {e}")
        return []

PROMPT_COMMONSENSE = """
Evaluate whether the knowledge graph triplet belongs to commonsense knowledge:
Source: <source>
Destination: <destination>
Relationship: <relationship>

Analysis Requirements:
1. Universality: Is this a basic fact known or agreed upon by the majority of people?
2. Stability: Is this relationship long-term stable and not subject to frequent changes?
3. Fundamentality: Is this about the basic attributes or relationships of entities, rather than detailed information?
4. Objectivity: Can this relationship be objectively verified without subjective judgment?

Scoring Guidelines:

- High-Scoring Examples (0.8-1.0):​​
•(Beijing, China, Beijing is the capital of China)
•(Sun, Earth, The Sun provides light and heat to the Earth)
•(Water, Hydrogen and Oxygen, Water is composed of hydrogen and oxygen)
•(Shakespeare, Hamlet, Shakespeare is the author of "Hamlet")

​​- Medium-Scoring Examples (0.5-0.7):​
•(Apple Inc., iPhone, Apple Inc. produces iPhone) - Business information, relatively stable
•(Eiffel Tower, Paris, The Eiffel Tower is in Paris) - Geographic information, widely known

- Low-Scoring Examples (0.0-0.4):
•(An actor, A movie, The actor's performance in a specific scene) - Too specific
•(A company, A product, The product's sales data in a specific year) - Timely information
•(A person, A place, The person went to a specific place yesterday) - Temporary event

Output Format:
Provide your response as a valid JSON object with the following structure:
{
    "analysis": "concise analysis",
    "score": 0.5
}

The score should be a float between 0.0-1.0 with two decimal precision.
"""

PROMPT_REASON = """
Evaluate the reasonableness of the knowledge graph triplet with precision:
Source: <source>
Destination: <destination>
Relationship: <relationship>

Analysis Requirements:
1. Semantic Accuracy: Evaluate if the relationship statement accurately describes the connection between the source and destination entities. Consider domain knowledge and factual correctness.
2. Relevance: Assess if the connection between these entities is meaningful and significant rather than trivial or coincidental.
3. Specificity: Determine if the relationship provides clear, specific information about how the entities are connected rather than being vague or overly general.
4. Logical Coherence: Check if the triplet follows proper semantic and syntactic patterns expected in knowledge graphs.
5. Entity Type Compatibility: Verify that the relationship makes sense given the types of entities involved (person-to-person, organization-to-event, etc.).

Scoring Guidelines:
- 0.0-0.3: Invalid or highly questionable relationship (factually wrong, illogical, or meaningless)
- 0.4-0.6: Partially valid but problematic (somewhat relevant but vague, imprecise, or contains minor inaccuracies)
- 0.7-0.8: Mostly valid (accurate but could be more specific or informative)
- 0.9-1.0: Fully valid (accurate, specific, informative, and logically sound)

Optimization Notes:
1. Focus on direct evaluation without unnecessary elaboration.
2. Use domain-specific reasoning where applicable.

Output Format:
Provide your response as a valid JSON object with the following structure:
{
    "analysis": "concise analysis",
    "score": 0.5
}

The score should be a float between 0.0-1.0 with two decimal precision.
"""

def get_model_score(client, edge):
    """获取单个边的模型评分"""
    source = edge.get('source', '')
    destination = edge.get('destination', '')
    relationship = edge.get('relationship', '')

    prompt = PROMPT_REASON.replace('<source>', source).replace('<destination>', destination).replace('<relationship>', relationship)

    max_retries = 3
    temperature = 0.0
    for retry in range(max_retries):
        try:
            model = MODEL_NAME or client.models.list().data[0].id
            if not MODEL_NAME:
                global PRINTED_MODEL
                if not PRINTED_MODEL:
                    print(f"[INFO] Auto-selected model: {model}")
                    PRINTED_MODEL = True
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a knowledge graph expert who evaluates whether the knowledge graph triplet belongs to commonsense knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

            content = response.choices[0].message.content

            json_response = json_repair.loads(content.strip())
            score = json_response.get('score')
            analysis = json_response.get('analysis', content)

            return {
                'source': source,
                'destination': destination,
                'relationship': relationship,
                'score': float(score) if score is not None else None,
                'analysis': analysis
            }

        except Exception as e:
            temperature += 0.3

def process_batch(edges_batch, client):
    """处理一批边"""
    results = []
    for edge in edges_batch:
        result = get_model_score(client, edge)
        results.append(result)
        time.sleep(RATE_LIMIT_DELAY)  # 添加小延迟避免API限制
    return results

async def process_edges_concurrent(input_file, output_file):
    """并发处理所有edge并获取评分"""
    
    # 加载GraphML数据
    print(f"正在加载GraphML文件: {input_file}")
    edges_data = load_graphml_edges(input_file)
    
    if not edges_data:
        print("没有找到边数据或文件加载失败")
        return
        
    print(f"共找到 {len(edges_data)} 条边")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 初始化输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # JSONL format doesn't need initial brackets
    
    total_edges = len(edges_data)
    processed_count = 0
    
    # 创建进度条
    pbar = tqdm(total=total_edges, desc="Processing edges")
    
    # 将边分成批次
    batches = [edges_data[i:i+BATCH_SIZE] for i in range(0, len(edges_data), BATCH_SIZE)]
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        loop = asyncio.get_event_loop()
        
        # 创建处理函数
        process_func = partial(process_batch, client=client)
        
        # 存储所有任务
        tasks = []
        for batch in batches:
            # 提交任务到线程池
            task = loop.run_in_executor(executor, process_func, batch)
            tasks.append(task)
        
        # 等待所有任务完成并处理结果
        for batch_idx, future in enumerate(asyncio.as_completed(tasks)):
            batch_results = await future
            
            # 写入结果到文件 (JSONL format)
            with open(output_file, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    json_str = json.dumps(result, ensure_ascii=False)
                    f.write(json_str + '\n')
            
            # 更新进度
            processed_count += len(batch_results)
            pbar.update(len(batch_results))
    
    # 关闭进度条
    pbar.close()
    
    print(f"处理完成，结果已保存到: {output_file}")
    
    # 读取结果文件进行统计 (JSONL format)
    scored_edges = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    edge_data = json.loads(line)
                    scored_edges.append(edge_data)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line: {line[:100]}...")

    # 输出统计信息
    if scored_edges:
        valid_scores = [edge['score'] for edge in scored_edges if edge['score'] is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"\n统计信息:")
            print(f"总边数: {len(scored_edges)}")
            print(f"有效评分数: {len(valid_scores)}")
            print(f"平均分数: {avg_score:.2f}")
            print(f"最高分数: {max(valid_scores):.2f}")
            print(f"最低分数: {min(valid_scores):.2f}")
        else:
            print(f"共处理了 {len(scored_edges)} 条边，但没有有效的评分")
    else:
        print("没有找到有效的边数据进行统计")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="Result/cs/rkg_graph/graph_storage/graph_storage_nx_data.graphml")
    parser.add_argument("--output_file", type=str, default="Result/cs/rkg_graph/graph_storage/graph_storage_nx_data_edge_reason.jsonl")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rate_limit_delay", type=float, default=0.1)
    args = parser.parse_args()

    INPUT_FILE = args.input_file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    OUTPUT_FILE = args.output_file
    MAX_WORKERS = args.max_workers
    BATCH_SIZE = args.batch_size
    RATE_LIMIT_DELAY = args.rate_limit_delay

    with open("./Option/Config2.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_url = config.get('llm', {}).get('llm_base_url')
    api_key = config.get('llm', {}).get('api_key')
    model_name = config.get('llm', {}).get('model')
    MODEL_NAME = model_name
    client = OpenAI( 
        base_url=base_url,  
        api_key=api_key,  
    )
    if MODEL_NAME:
        print(f"[INFO] Using configured model: {MODEL_NAME}")

    asyncio.run(process_edges_concurrent(INPUT_FILE, OUTPUT_FILE)) 