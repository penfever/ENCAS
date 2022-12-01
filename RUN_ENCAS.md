### CPU

singularity exec \
  --overlay /scratch/bf996/singularity_containers/encas_env.ext3:ro \
  --overlay /scratch/bf996/datasets/in100.sqf:ro \
  --overlay /scratch/bf996/datasets/laion100.sqf:ro \
  --overlay /scratch/bf996/datasets/openimages100.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-r.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-a.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash

source /ext3/env.sh

### GPU

singularity \
  exec $nv \
  --overlay /scratch/bf996/singularity_containers/encas_env.ext3:ro \
  --overlay /scratch/bf996/datasets/in100.sqf:ro \
  --overlay /scratch/bf996/datasets/laion100.sqf:ro \
  --overlay /scratch/bf996/datasets/openimages100.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-r.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-a.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash

source /ext3/env.sh

## Extras

cuda11.3.0-cudnn8-devel-ubuntu20.04.sif
/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

## Build MO-GOMEA

export LD_LIBRARY_PATH="/ext3/miniconda3/lib/"

export LIBRARY_PATH="/ext3/miniconda3/lib/"

In CMakeLists.txt change target_include_directories to refer to the header files of your Python distribution.

/usr/bin/cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" /scratch/bf996/MO-GOMEA-for-ENCAS/ -B/scratch/bf996/MO-GOMEA-for-ENCAS/cmake-build-release-remote

/usr/bin/cmake --build /scratch/bf996/MO-GOMEA-for-ENCAS/cmake-build-release-remote --target MO_GOMEA -- -j 6

## Env update

conda env update --file /scratch/bf996/encas/data/env_encas.yml --prune

export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

pip install inplace-abn

pip install git+https://github.com/AwesomeLemon/once-for-all

pip install git+https://github.com/AwesomeLemon/pySOT

## reproduce nat

python nat_run_many.py --config configs_nat/cifar10_reproducenat.yml

### error

A process in the process pool was terminated abruptly while the future was running or pending.
Setting random seed to 1728
self.path_logs='/scratch/bf996/encas/logs/cifar10_reproducenat/search_algo:nsga3!subset_selector:reference/5'
supernet_paths=['data/ofa/supernet_w1.0', 'data/ofa/supernet_w1.2']
Color jitter: None, resize_scale: 0.08, img_size: 256
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
self.ref_dirs.shape=(100, 2)
Lock 22558123689440 released on /scratch/bf996/encas/logs/cifar10_reproducenat/gpu_0.lockLock 22918375038832 acquired on /scratch/bf996/encas/logs/cifar10_reproducenat/gpu_0.lockpercent_w1_0=0.5
self.ref_dirs.shape=(100, 2)
Traceback (most recent call last):
  File "/scratch/bf996/encas/nat_run_many.py", line 128, in execute_run
    main(algo_run_kwargs)
  File "/scratch/bf996/encas/nat.py", line 958, in main
    engine.search()
  File "/scratch/bf996/encas/nat.py", line 224, in search
    archive, first_iteration = self.create_or_restore_archive(ref_pt)
  File "/scratch/bf996/encas/nat.py", line 291, in create_or_restore_archive
    objs_evaluated = self.train_and_evaluate(arch_doe, 0)
  File "/scratch/bf996/encas/nat.py", line 405, in train_and_evaluate
    self._train(archs, it, n_epochs=n_epochs, if_warmup=if_warmup)
  File "/scratch/bf996/encas/nat.py", line 460, in _train
    future.result()
  File "/ext3/miniconda3/envs/env_encas/lib/python3.8/concurrent/futures/_base.py", line 439, in result
    return self.__get_result()
  File "/ext3/miniconda3/envs/env_encas/lib/python3.8/concurrent/futures/_base.py", line 388, in __get_result
    raise self._exception
concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

A process in the process pool was terminated abruptly while the future was running or pending.