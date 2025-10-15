dataset="${dataset:-cs}"
method="${method:-LGraphRAG}"
index_name="${index_name:-lkg_graph}"
node_reduction="${node_reduction:-0.40}"
edge_threshold="${edge_threshold:-0.00}"
llm_base_url="${llm_base_url:-https://openrouter.ai/api/v1}"
embedding_base_url="${embedding_base_url:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

cluster_type=kmeans # kmeans/entity_block/structure
embedding_type=llm # llm/TransE/DistMult/ComplEx/RGCN/CompGCN
similarity_mode=node_only # node_only/neighbor_only/node_neighbor/neisubset_only/node_neisubset
merge_type=reduction_only # reduction_only/reduction_synonym/synonym_only/random_deletion

exp_name="${dataset}_${node_reduction}_${edge_threshold}_${cluster_type}_${embedding_type}_${similarity_mode}_${merge_type}"

echo "dataset: ${dataset}"
echo "method: ${method}"
echo "index_name: ${index_name}"
echo "node_reduction: ${node_reduction}"
echo "edge_threshold: ${edge_threshold}"
echo "llm_base_url: ${llm_base_url}"
echo "embedding_base_url: ${embedding_base_url}"
echo "cluster_type: ${cluster_type}"
echo "embedding_type: ${embedding_type}"
echo "similarity_mode: ${similarity_mode}"
echo "merge_type: ${merge_type}"
echo "exp_name: ${exp_name}"

### Step 0: Initialize the KG
if [ -f "Result/${dataset}/default_experiment/${index_name}/node_only/Results/results.json" ]; then
    echo "Skipping Step 0: KG already initialized"
else
    # python main.py --opt Option/Method/${method}.yaml --dataset_name ${dataset}
    python main.py --opt Option/Method/${method}.yaml --dataset_name ${dataset} --llm_base_url "${llm_base_url:-https://openrouter.ai/api/v1}"
fi

### Step 1: Cluster nodes
k=10
batch_size=10
graphml_path="Result/${dataset}/${index_name}/graph_storage/graph_storage_nx_data.graphml"
k_means_path="Result/${dataset}/${index_name}/k_means/${cluster_type}"
orig_text_path="${k_means_path}/${dataset}_original_texts.json"
orig_emb_path="${k_means_path}/${dataset}_original_embeddings.npy"
output_path="${k_means_path}/${dataset}_${cluster_type}_similar_nodes.json"
cluster_map_path="${k_means_path}/${dataset}_${cluster_type}_cluster_map.json"

if [ -f "$orig_text_path" ] && [ -f "$orig_emb_path" ] && [ -f "$output_path" ] && [ -f "$cluster_map_path" ]; then
    echo "Skipping Step 1: All cluster files already exist"
else
    python denoise_kg/cluster.py \
        --dataset $dataset \
        --k $k \
        --batch_size $batch_size \
        --graphml_path $graphml_path \
        --k_means_path $k_means_path \
        --orig_text_path $orig_text_path \
        --orig_emb_path $orig_emb_path \
        --output_path $output_path \
        --cluster_map_path $cluster_map_path \
        --cluster_type $cluster_type
fi

### Step 2.1: Similar nodes
max_neighbors=10
batch_size=10
node_weight=0.5
neighbor_weight=0.5
output_dir="Result/${dataset}/${index_name}/node_neighbors/${cluster_type}/${embedding_type}"
export similar_nodes_path="${output_dir}/${exp_name}.json"

if [ -f $similar_nodes_path ]; then
    echo "Skipping Step 2.1: Similar nodes already calculated"
else
    python denoise_kg/similar_test.py \
        --dataset_name $dataset \
        --similarity_mode $similarity_mode \
        --target_reduction_percent $node_reduction \
        --max_neighbors $max_neighbors \
        --batch_size $batch_size \
        --node_weight $node_weight \
        --neighbor_weight $neighbor_weight \
        --graphml_path $graphml_path \
        --node_data_path $output_path \
        --cluster_map_path $cluster_map_path \
        --similar_nodes_path $similar_nodes_path \
        --cache_dir $k_means_path \
        --embedding_type $embedding_type
fi

### Step 2.2: Edge clean
edge_clean_path="Result/${dataset}/${index_name}/edge_process/edge_reason_${edge_threshold}.jsonl"
if [ -f $edge_clean_path ]; then
    echo "Skipping Step 2.2: Edge clean already calculated"
else
    python denoise_kg/edge_clean.py --input_file $graphml_path --output_file $edge_clean_path
fi

### Step 3: Merge nodes
new_graphml_name="graph_storage_nx_data_${exp_name}.graphml"
output_path="Result/${dataset}/${index_name}/graph_storage/${new_graphml_name}"

if [ -f $output_path ]; then
    echo "Skipping Step 3: Merged graph already exists"
else
    python denoise_kg/merge.py \
        --dataset_name $dataset \
        --graph_path $graphml_path \
        --similar_nodes_path $similar_nodes_path \
        --output_path $output_path \
        --threshold_edge_reason $edge_threshold \
        --merge_type $merge_type \
        --similarity $node_reduction
fi

### Step 4: RAG response
if [ ! -f "Result/${dataset}/${index_name}/graph_storage/${new_graphml_name}" ]; then
    echo "File ${new_graphml_name} does not exist. Exiting."
    exit 1
fi
path_entities_vdb="Result/${dataset}/${index_name}/entities_vdb"
path_relations_vdb="Result/${dataset}/${index_name}/relations_vdb"
rm -rf $path_entities_vdb
rm -rf $path_relations_vdb

python main.py \
    --opt Option/Method/${method}.yaml \
    --dataset_name $dataset \
    --graph_file_name $new_graphml_name \
    --exp_name $exp_name \
    --llm_base_url $llm_base_url \
    --embedding_base_url $embedding_base_url

### Step 5: Winrate
file_2_path="Result/${dataset}/default_experiment/${index_name}/node_only/Results/results.json"
file_1_path="Result/${dataset}/${exp_name}/${index_name}/node_only/Results/results.json"
output_dir="Result/${dataset}/${index_name}/evaluation_outputs/${exp_name}"
config_path="Option/Config2.yaml"

python denoise_kg/winrate.py \
    --file_1_path $file_1_path \
    --file_2_path $file_2_path \
    --output_dir $output_dir \
    --config_path $config_path