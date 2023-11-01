import os
import torch
from models.GraphMixer import GraphMixer
from utils.utils import convert_to_gpu
from utils.utils import get_neighbor_sampler
from utils.DataLoader import  get_node_classification_data_inference
from utils.load_configs import get_node_classification_args
from tqdm import tqdm
from ray_processing import TorchPredictor
import ray

if __name__ == "__main__":

    args = get_node_classification_args()
    node_raw_features, edge_raw_features, full_data, graph_df = \
        get_node_classification_data_inference(dataset_name=args.dataset_name)

    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    
    # train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(full_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    
    model = GraphMixer(args, node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    

    args.seed = 0
    args.load_model_name = f'{args.model_name}_seed{args.seed}'
    args.save_model_name = f'node_classification_{args.model_name}_seed{args.seed}'

    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
    args.save_model_name = f'node_classification_{args.model_name}_seed{args.seed}'
    save_model_path = os.path.join(save_model_folder, f"{args.save_model_name}.pkl")
    model = convert_to_gpu(model, device=args.device)
    model.load_state_dict(torch.load(save_model_path))
    ds = ray.data.from_pandas(graph_df)
    results = ds.map_batches(TorchPredictor,
                             num_gpus=1,
                             batch_size=1024,
                             compute=ray.data.ActorPoolStrategy(min_size=1, max_size=5),
                             fn_constructor_args=(model,))
    save_result_folder = f"./saved_results_predictions/{args.model_name}/{args.dataset_name}"
    results.write_csv(save_result_folder)
    print("succeed")
