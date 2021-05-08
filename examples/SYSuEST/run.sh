#!/bin/bash
#SBATCH -J WuK
#SBATCH -p gpu_v100
#SBATCH -N 1
#SBATCH --exclusive

SYSUEST_HOME=/mnt/pan/users/WuK/SYSuEST

spack unload -a
spack load cmake@3.18.3
spack load gcc@7.5.0

if true; then
    spack load openmpi%intel+cuda fabrics=ucx ^ucx+cuda+gdrcopy
else
    spack load mvapich2%intel+cuda
    # http://mvapich.cse.ohio-state.edu/static/media/mvapich/mvapich2-2.3.4-userguide.html#x1-30300011.121
    export MV2_USE_CUDA=1
    export MV2_CUDA_NONBLOCKING_STREAMS=1
    export MV2_CUDA_BLOCK_SIZE=524288
    export MV2_CUDA_KERNEL_VECTOR_TIDBLK_SIZE=1024
    export MV2_CUDA_IPC=1
    export MV2_CUDA_SMP_IPC=0
    export MV2_SMP_USE_CMA=1
fi

spack find -v --loaded

for CB in 12 11 10; do
for WORKLOAD in random GHZ GHZ_QFT_N; do
    if [ $WORKLOAD == "GHZ" ]; then
        USER_SOURCE="$SYSUEST_HOME/examples/SYSuEST/GHZ_QFT.c"
    else
        USER_SOURCE="$SYSUEST_HOME/examples/SYSuEST/$WORKLOAD.c"
    fi
    mkdir -p "$SYSUEST_HOME/../SYSuEST_build_$WORKLOAD"
    cd "$SYSUEST_HOME/../SYSuEST_build_$WORKLOAD"
    rm -fr *
    cmake \
        -DUSER_SOURCE=$USER_SOURCE \
        -DCMAKE_C_FLAGS=" -DCOMBINE_BIT=$CB -lcublas -lcudart " \
        -DDISTRIBUTED=1 \
        -DGPUACCELERATED=1 \
        -DGPU_COMPUTE_CAPABILITY=80 \
        "$SYSUEST_HOME"
    make -j
    for N in 4 2 1; do
        echo "Executing $WORKLOAD with $N process..."
        `which mpirun` -x LD_LIBRARY_PATH -n $N --rankfile $SYSUEST_HOME/examples/SYSuEST/rankfile ./demo
        for FILENAME in "probs.dat" "stateVector.dat"; do
            echo "Checking $FILENAME..."
            diff $FILENAME "$SYSUEST_HOME/examples/SYSuEST/${FILENAME}_${WORKLOAD}"
        done
    done
done
done
