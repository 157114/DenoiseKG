from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator
from tqdm.asyncio import tqdm_asyncio
import sys
import csv

# Increase the CSV field size limit
csv.field_size_limit(2147483647)



def check_dirs(opt):
    method_name = opt.index_name
    similarity_folder = opt.graph.similarity_mode
    # For each query, save the results in a separate directory
    result_dir = os.path.join(opt.working_dir, opt.exp_name, method_name, similarity_folder, "Results")
    # Save the current used config in a separate directory
    config_dir = os.path.join(opt.working_dir, opt.exp_name, method_name, similarity_folder, "Configs")
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, method_name, similarity_folder, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = args.opt[args.opt.rindex("/") + 1 :]
    basic_name = os.path.join(args.opt.split("/")[0], "Config2.yaml")
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    return result_dir


async def run_single_query(semaphore, digimon, query_item):
    """
    A helper async function to run one query while respecting the semaphore.
    """
    async with semaphore:
        question = query_item["question"]
        result_output = await digimon.query(question)
        query_item["output"] = result_output
        return query_item

async def concurrent_wrapper_query(query_dataset, digimon, result_dir):
    """
    Processes the entire query dataset concurrently.
    """
    # Define how many queries to run at the same time.
    # Start with a safe number like 5 or 8 to avoid API rate limits.
    CONCURRENCY_LIMIT = 64
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    print(f"--- Starting concurrent query processing (limit: {CONCURRENCY_LIMIT}) ---")

    # Create a list of tasks to run
    tasks = []
    for i in range(len(query_dataset)):
        query_item = query_dataset[i]
        tasks.append(run_single_query(semaphore, digimon, query_item))
    
    # Run the tasks concurrently and show a progress bar
    all_res = await tqdm_asyncio.gather(*tasks)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    print(f"\n✅ All queries processed. Results saved to {save_path}")
    return save_path

async def wrapper_evaluation(path, opt, result_dir):
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))


if __name__ == "__main__":

    # with open("./book.txt") as f:
    #     doc = f.read()

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="Option/Method/LightRAG.yaml", help="Path to option YMAL file.")
    parser.add_argument("--dataset_name", type=str, default="mini_cs", help="Name of the dataset.")
    parser.add_argument("--graph_file_name", type=str, default="graph_storage_nx_data.graphml", help="Name of the graph file.")
    parser.add_argument("--llm_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Name of the graph file.")
    parser.add_argument("--embedding_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Name of the graph file.")
    parser.add_argument("--exp_name", type=str, default="default_experiment", help="Name of the experiment.")
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    if args.graph_file_name: opt.graph.graph_file_name = args.graph_file_name
    if args.exp_name: opt.exp_name = args.exp_name
    if args.llm_base_url: opt.llm.base_url = args.llm_base_url
    if args.embedding_base_url: opt.embedding.base_url = args.embedding_base_url
    digimon = GraphRAG(config=opt)
    result_dir = check_dirs(opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    async def main_async():
        corpus = query_dataset.get_corpus()
        await digimon.insert(corpus)
        
        # Run the concurrent querying
        save_path = await concurrent_wrapper_query(query_dataset, digimon, result_dir)
        
        # Run the evaluation
        await wrapper_evaluation(save_path, opt, result_dir)

    # Use a single asyncio.run() to execute the entire async workflow
    asyncio.run(main_async())

    # for train_item in dataloader:

    # a = asyncio.run(digimon.query("Who is Fred Gehrke?"))

    # asyncio.run(digimon.query("Who is Scrooge?"))
