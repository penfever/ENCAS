# Encas

singularity \
  exec --nv \
  --overlay /scratch/bf996/singularity_containers/encas_env.ext3:ro \
  --overlay /scratch/bf996/datasets/in100.sqf:ro \
  --overlay /scratch/bf996/datasets/laion100.sqf:ro \
  --overlay /scratch/bf996/datasets/openimages100.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-r.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-a.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
  /bin/bash

source /ext3/env.sh

## Extras
cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif
cuda11.3.0-cudnn8-devel-ubuntu20.04.sif
/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

## imagenet-v2

rsync -aP /scratch/bf996/datasets/imagenetv2-top-images-format-val/ /scratch/bf996/datasets/imagenetv2-all/

find /scratch/bf996/datasets/imagenetv2-all -type f | wc -l (should be 20683)

## Build MO-GOMEA

export LD_LIBRARY_PATH="/ext3/miniconda3/lib/"

export LIBRARY_PATH="/ext3/miniconda3/lib/"

In CMakeLists.txt change target_include_directories to refer to the header files of your Python distribution.

/usr/bin/cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" /scratch/bf996/MO-GOMEA-for-ENCAS/ -B/scratch/bf996/MO-GOMEA-for-ENCAS/cmake-build-release-remote

/usr/bin/cmake --build /scratch/bf996/MO-GOMEA-for-ENCAS/cmake-build-release-remote --target MO_GOMEA -- -j 6

## Env update

<!-- conda create --name env_encas python=3.8

conda install -c "nvidia/label/cuda-11.6.1" cuda-toolkit -->

<!-- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 -->

conda env update --file /scratch/bf996/encas/data/env_encas.yml --prune

conda activate env_encas

export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

pip install inplace-abn

pip install git+https://github.com/AwesomeLemon/once-for-all

pip install git+https://github.com/AwesomeLemon/pySOT

python data_utils/store_labels.py

## reproduce nat

python nat_run_many.py --config configs_nat/cifar10_testrun.yml

## params

default_kwargs = {
    'experiment_name': ['debug_run', 'location of dir to save'],
    'resume': [None, 'resume search from a checkpoint'],
    'sec_obj': ['flops', 'second objective to optimize simultaneously'],
    'iterations': [30, 'number of search iterations'],
    'n_doe': [100, 'number of architectures to sample initially '
                   '(I kept the old name which is a bit weird; "doe"=="design of experiment")'],
    'n_iter': [8, 'number of architectures to evaluate in each iteration'],
    'predictor': ['rbf', 'which accuracy predictor model to fit'],
    'data': ['/export/scratch3/aleksand/data/CIFAR/', 'location of the data corpus'],
    'dataset': ['cifar10', 'name of the dataset [imagenet, cifar10, cifar100, ...]'],
    'n_workers': [8, 'number of workers for dataloaders'],
    'vld_size': [10000, 'validation size'],
    'total_size': [None, 'train+validation size'],
    'trn_batch_size': [96, 'train batch size'],
    'vld_batch_size': [96, 'validation batch size '],
    'n_epochs': [5, 'test batch size for inference'],
    'supernet_path': [['/export/scratch3/aleksand/nsganetv2/data/ofa_mbv3_d234_e346_k357_w1.0'], 'list of paths to supernets'],
    'search_algo': ['nsga3', 'which search algo to use [NSGA-III, MO-GOMEA, random]'],
    'subset_selector': ['reference', 'which subset selector algo to use'],
    'init_with_nd_front_size': [0, 'initialize the search algorithm with subset of non-dominated front of this size'],
    'dont_check_duplicates': [False, 'if disable check for duplicates in search results'],
    'add_archive_to_candidates': [False, 'if a searcher should append archive to the candidates'],
    'sample_configs_to_train': [False, 'if instead of training selected candidates, a probability distribution '
                                       'should be constructed from archive, and sampled from (like in NAT)'],
    'random_seed': [42, 'random seed'],
    'n_warmup_epochs': [0, 'number of epochs for warmup'],
    'path_logs': ['/export/scratch3/aleksand/nsganetv2/logs/', 'Path to the logs folder'],
    'n_surrogate_evals': [800, 'Number of evaluations of the surrogate per meta-iteration'],
    'config_msunas_path': [None, 'Path to the yml file with all the parameters'],
    'gomea_exe': [None, 'Path to the mo-gomea executable file'],
    'alphabet': [['2'], 'Paths to text files (one per supernetwork) with alphabet size per variable'],
    'search_space': [['ensemble'], 'Supernetwork search space to use'],
    'store_checkpoint_freq': [1, 'Checkpoints will be stored for every x-th iteration'],
    'init_lr': [None, 'initial learning rate'],
    'ensemble_ss_names': [[], 'names of search spaces used in the ensemble'],
    'rbf_ensemble_size': [500, 'number of the predictors in the rbf_ensemble surrogate'],
    'cutout_size': [32, 'Cutout size. 0 == disabled'],
    'label_smoothing': [0.0, 'label smoothing coeff when doing classification'],
    'if_amp': [False, 'if train in mixed precision'],
    'use_gradient_checkpointing': [False, 'if use gradient checkpointing'],
    'lr_schedule_type': ['cosine', 'learning rate schedule; "cosine" is cyclic'],
    'if_cutmix': [False, 'if to use cutmix'],
    'weight_decay': [4e-5, ''],
    'if_center_crop': [True, 'if do center crop, or just resize to target size'],
    'auto_augment': ['rand-m9-mstd0.5', 'randaugment policy to use, or None to not use randaugment'],
    'resize_scale': [0.08, 'minimum resize scale in RandomResizedCrop, or None to not use RandomResizedCrop'],
    'search_goal': ['ensemble', 'Either "reproduce_nat" for reproducing NAT, or "ensemble" for everything else'],
    n_runs: [1, 'Not sure but I think this is for repeating the entire experiment multiple times with different random seeds']
}

## nsga3 wrapper

```
self.n_obj = 2 (number of objectives to optimize)
self.pop_size = 100
self.n_gen = n_evals // self.pop_size (n_evals is the number of archs to evaluate, O(10000) in the NAT paper)
self.ref_dirs = get_reference_directions("riesz", self.n_obj, 100) (100 "riesz" rays are used in NSGA3)

```

## next steps

* Figure out what objects are being compared in spearman rho
* concretize a few examples of those objects
* move the concrete examples to a notebook / Numpy environment
* Do the same with the frontier of the accuracy predictor
* Get anything else we need
* Figure out a way to solve the multi objective search problem in a notebook
* MHA (multi-heuristic A*) is a good candidate: https://github.com/mohakbhardwaj/planning_python/blob/master/planning_python/planners/MHAstar.py
* Graph edit distance? Simrank similarity? Laplacian matrix difference? https://networkx.org/documentation/stable/reference/algorithms/similarity.html https://www.wikiwand.com/en/Graph_edit_distance https://github.com/caesar0301/graphsim https://wadsashika.wordpress.com/2014/09/19/measuring-graph-similarity-using-neighbor-matching/

## notes

            # hull = ConvexHull(I)
            #TODO: initialize dists to heuristic values
            dists = []
            #TODO: compute true distance as euclidean dist to 
            for a in range(len(objs_to_examine)):
                pointwise_d = np.inf
                for i in range(len(hull.vertices)):
                    _, d = min_distance(I[hull.vertices[i]], I[hull.vertices[(i+1)%len(hull.vertices)]], pt)
                    if d < pointwise_d:
                        pointwise_d = d

            S = niching(objs[indices_selected][last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
        def min_distance(pt1, pt2, p):
        """ return the projection of point p (and the distance) on the closest edge formed by the two points pt1 and pt2"""
        l = np.sum((pt2-pt1)**2) ## compute the squared distance between the 2 vertices
        t = np.max([0., np.min([1., np.dot(p-pt1, pt2-pt1) /l])]) # I let the answer of question 849211 explains this
        proj = pt1 + t*(pt2-pt1) ## project the point
        return proj, np.sum((proj-p)**2) ## return the projection and the point

## Algo output

(env_encas) Singularity> python /scratch/bf996/encas/subset_selectors/astar_subset_selector.py
non dominated is [10 11], number of fronts is 2
ideal point is [-2. -3.], nadir is [ 1. -1.], worst is [4. 4.]
concat result: 
[10 11  9 13 14]
indices selected after concat
[False False False False False False False False False  True  True  True
 False  True  True False False False]
n remaining is 1
dists is [0.0, 0.0, 0.0]
Survivors are [0 1 0]
Selection complete: Selected 2 indices from 18 possibilities
Indices selected:
[False False False False False False False False False False  True  True
 False False False False False False]
Vertices selected (by objectives): 
[[-2. -1.]
 [ 1. -3.]]

## Pathing for plotting

nat_config_path = os.path.join(nsga_logs_path, experiment_path, 'config_msunas.yml')

def swa_for_whole_experiment(experiment_name, iters, supernet_name_in, target_runs=None):
    nsga_logs_path = utils.NAT_LOGS_PATH
    experiment_path = os.path.join(nsga_logs_path, experiment_name)

NAT_PATH = '/scratch/bf996/encas'
NAT_LOGS_PATH = os.path.join(NAT_PATH, 'logs')
NAT_DATA_PATH = os.path.join(NAT_PATH, 'data')