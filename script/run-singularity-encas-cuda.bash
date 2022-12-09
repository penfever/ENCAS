#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ "$(hostname -s)" =~ ^g[r,v,a] ]]; then nv="--nv"; fi

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
    /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"