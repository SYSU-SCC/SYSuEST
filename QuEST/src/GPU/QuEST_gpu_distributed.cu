// Distributed under MIT licence. See
// https://github.com/SYSU_SCC/SYSuEST/blob/master/LICENCE.txt for details

/** @file
 * An implementation of the backend in ../QuEST_internal.h for an MPI
 * environment. Distributed GPU acceleration is supported by Student Cluster
 * Competition Team @ Sun Yat-sen University.
 */

#include <cublas_v2.h>
#include <limits>
#include <mpi.h>
#include <stdio.h>
#include <vector>

#include "QuEST.h"
#include "QuEST_internal.h" // purely to resolve getQuESTDefaultSeedKey
#include "QuEST_precision.h"
#include "mt19937ar.h"

#if !defined(THREADS_PER_CUDA_BLOCK)
#define THREADS_PER_CUDA_BLOCK 1024
#endif

#if !defined(COMBINE_BIT)
#define COMBINE_BIT 10 // 11,12 for A100
#endif

#if !defined(EXCHANGE_WARMUP_BIT)
#define EXCHANGE_WARMUP_BIT 23
#endif

#if !defined(BUFFER_AMP)
#define BUFFER_AMP (1LL << EXCHANGE_WARMUP_BIT)
#endif

#ifdef USE_BLASPP

#include <blas.hh>

#else

namespace blas {
static const int DEV_QUEUE_FORK_SIZE = 15;

void set_device(int device) { cudaSetDevice(device); }

void stream_create(cudaStream_t *stream) {
  cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
}

void stream_destroy(cudaStream_t stream) { cudaStreamDestroy(stream); }

void stream_synchronize(cudaStream_t stream) { cudaStreamSynchronize(stream); }

void handle_create(cublasHandle_t *handle) { cublasCreate(handle); }

void handle_destroy(cublasHandle_t handle) { cublasDestroy(handle); }

void handle_set_stream(cublasHandle_t handle, cudaStream_t stream) {
  cublasSetStream(handle, stream);
}

void event_create(cudaEvent_t *event) {
  cudaEventCreateWithFlags(event, cudaEventDisableTiming);
}

void event_destroy(cudaEvent_t event) { cudaEventDestroy(event); }

void event_record(cudaEvent_t event, cudaStream_t stream) {
  cudaEventRecord(event, stream);
}

void stream_wait_event(cudaStream_t stream, cudaEvent_t event,
                       unsigned int flags) {
  cudaStreamWaitEvent(stream, event, flags);
}

class Queue {
private:
  int device_;
  size_t current_stream_index_;
  size_t num_active_streams_;

  // associated device blas handle
  cublasHandle_t handle_;

  // pointer to current stream (default or fork mode)
  cudaStream_t *current_stream_;

  // default CUDA stream for this queue; may be NULL
  cudaStream_t default_stream_;

  // parallel streams in fork mode
  cudaEvent_t parallel_events_[DEV_QUEUE_FORK_SIZE];

  cudaEvent_t default_event_;

  cudaStream_t parallel_streams_[DEV_QUEUE_FORK_SIZE];

public:
  Queue(int device, int64_t batch_size) {
    device_ = device;
    set_device(device_);
    stream_create(&default_stream_);
    handle_create(&handle_);
    handle_set_stream(handle_, default_stream_);
    current_stream_ = &default_stream_;
    num_active_streams_ = 1;
    current_stream_index_ = 0;

    // create parallel streams
    for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
      stream_create(&parallel_streams_[i]);
    }

    // create default and parallel events
    event_create(&default_event_);
    for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
      event_create(&parallel_events_[i]);
    }
  }
  ~Queue() {
    handle_destroy(handle_);
    stream_destroy(default_stream_);

    // destroy parallel streams
    for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
      stream_destroy(parallel_streams_[i]);
    }

    // destroy events
    event_destroy(default_event_);
    for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
      event_destroy(parallel_events_[i]);
    }
  }
  void sync() {
    if (current_stream_ == &default_stream_) {
      stream_synchronize(default_stream_);
    } else {
      for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
        stream_synchronize(parallel_streams_[i]);
      }
    }
  }
  void fork() {
    // check if queue is already in fork mode
    if (current_stream_ != &default_stream_)
      return;

    // make sure dependencies are respected
    event_record(default_event_, default_stream_);
    for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
      stream_wait_event(parallel_streams_[i], default_event_, 0);
    }

    // assign current stream
    current_stream_index_ = 0;
    num_active_streams_ = DEV_QUEUE_FORK_SIZE;
    current_stream_ = &parallel_streams_[current_stream_index_];

    // assign cublas handle to current stream
    handle_set_stream(handle_, *current_stream_);
  }
  void join() {
    if (current_stream_ == &default_stream_)
      return;

    // make sure dependencies are respected
    for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
      event_record(parallel_events_[i], parallel_streams_[i]);
      stream_wait_event(default_stream_, parallel_events_[i], 0);
    }

    // assign current stream
    current_stream_index_ = 0;
    num_active_streams_ = 1;
    current_stream_ = &default_stream_;

    // assign current stream to blas handle
    handle_set_stream(handle_, *current_stream_);
  }
  void revolve() {
    // return if not in fork mode
    if (current_stream_ == &default_stream_)
      return;

    // choose the next-in-line stream
    current_stream_index_ = (current_stream_index_ + 1) % num_active_streams_;
    current_stream_ = &parallel_streams_[current_stream_index_];

    // assign current stream to blas handle
    handle_set_stream(handle_, *current_stream_);
  }
  auto stream() const { return *current_stream_; }
  auto handle() const { return handle_; }
};
}; // namespace blas
#endif

class wukDeviceHandle : public blas::Queue {
private:
  char *deviceMemPool, *hostMemPool;
  int64_t deviceMemPoolSize, hostMemPoolSize;
  std::vector<int64_t> deviceMemPoolSizeBak, hostMemPoolSizeBak;
  size_t current_stream_index_;
  size_t num_active_streams_;

public:
  wukDeviceHandle(int device)
      : blas::Queue(device, 0), num_active_streams_(1),
        current_stream_index_(0) {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cublasSetPointerMode(handle(), CUBLAS_POINTER_MODE_HOST);
    size_t total, avail;
    cudaMemGetInfo(&avail, &total);
    hostMemPoolSize = deviceMemPoolSize =
        static_cast<size_t>(avail * 0.925 + (1 << 8) - 1) >> 8 << 8;
    cudaMalloc(&deviceMemPool, deviceMemPoolSize);
    // check gpu memory allocation was successful
    if (!deviceMemPool) {
      printf("Could not allocate memory on Device!\n");
      exit(EXIT_FAILURE);
    }
#if __CUDA_API_VERSION >= 11000
    size_t workspaceSizeInBytes = 1LL << 23; // 8MB
    cublasSetWorkspace(handle(), device_malloc<char>(workspaceSizeInBytes),
                       workspaceSizeInBytes);
#endif
#if defined(uSE_PINNED_MEMORY)
    cudaHostAlloc(&hostMemPool, hostMemPoolSize, cudaHostAllocPortable);
#else
    hostMemPool = new char[hostMemPoolSize];
#endif
    if (!hostMemPool) {
      printf("Could not allocate memory on Host!\n");
      exit(EXIT_FAILURE);
    }
  }
  ~wukDeviceHandle() {
#if defined(USE_PINNED_MEMORY)
    cudaFreeHost(hostMemPool);
#else
    delete[] hostMemPool;
#endif
    cudaFree(deviceMemPool);
  }
  void fork() {
    blas::Queue::fork();
    if (num_active_streams_ != blas::DEV_QUEUE_FORK_SIZE) {
      num_active_streams_ = blas::DEV_QUEUE_FORK_SIZE;
      current_stream_index_ = 0;
    }
  }
  void join() {
    blas::Queue::join();
    if (num_active_streams_ != 1) {
      num_active_streams_ = 1;
      current_stream_index_ = 0;
    }
  }
  void revolve() {
    blas::Queue::revolve();
    current_stream_index_ = (current_stream_index_ + 1) % num_active_streams_;
  }
  auto get_current_stream_index() const { return current_stream_index_; }
  auto get_num_active_streams() const { return num_active_streams_; }
  void device_store() { deviceMemPoolSizeBak.push_back(deviceMemPoolSize); }
  void device_store_pinned() { hostMemPoolSizeBak.push_back(hostMemPoolSize); }
  void device_restore() {
    if (!deviceMemPoolSizeBak.empty()) {
      deviceMemPoolSize = deviceMemPoolSizeBak.back();
      deviceMemPoolSizeBak.pop_back();
    }
  }
  void device_restore_pinned() {
    if (!hostMemPoolSizeBak.empty()) {
      hostMemPoolSize = hostMemPoolSizeBak.back();
      hostMemPoolSizeBak.pop_back();
    }
  }
  template <typename T> T *device_malloc(int64_t nelements) {
    int64_t size = nelements * sizeof(T) + (1 << 8) - 1 >> 8 << 8;
    if (this->deviceMemPoolSize < size) {
      printf("Device buffer is not enough!\n");
      exit(EXIT_FAILURE);
    }
    this->deviceMemPoolSize -= size;
    return reinterpret_cast<T *>(this->deviceMemPool + this->deviceMemPoolSize);
  }
  template <typename T> T *device_malloc_pinned(int64_t nelements) {
    int64_t size = nelements * sizeof(T) + (1 << 8) - 1 >> 8 << 8;
    if (this->hostMemPoolSize < size) {
      printf("Host buffer is not enough!\n");
      exit(EXIT_FAILURE);
    }
    this->hostMemPoolSize -= size;
    return reinterpret_cast<T *>(this->hostMemPool + this->hostMemPoolSize);
  }
  double asum(int64_t n, const double *x, int64_t incx) {
    const int64_t fn = std::numeric_limits<int>::max();
    if (n > fn)
      return asum(fn, x, incx) + asum(n - fn, x + fn * incx, incx);
    double r = 0;
    cublasDasum(handle(), n, x, incx, &r);
    sync();
    return r;
  }
  double dot(int64_t n, const double *x, int64_t incx, const double *y,
             int64_t incy) {
    const int64_t fn = std::numeric_limits<int>::max();
    if (n > fn)
      return dot(fn, x, incx, y, incy) +
             dot(n - fn, x + fn * incx, incx, y + fn * incy, incy);
    double r = 0;
    cublasDdot(handle(), n, x, incx, y, incy, &r);
    sync();
    return r;
  }
};

typedef enum {
  Qfunc_statevec_multiControlledUnitary,
  Qfunc_statevec_multiControlledPhaseShift
} Qfunc;

typedef struct {
  union {
    struct {
      cuDoubleComplex u[2][2];
      long long int mask;
      int targetQubit;
    } statevec_multiControlledUnitary;
    struct {
      cuDoubleComplex term;
      long long int mask, ctrlFlipMask;
    } statevec_multiControlledPhaseShift;
  } args;
  Qfunc func;
} Qtask;

typedef struct {
  QuESTEnv env;
  std::vector<Qtask> tbd;
  std::vector<qreal> prob;
} Rhandle;

static wukDeviceHandle *get_wukDeviceHandle(Qureg qureg) {
  return reinterpret_cast<wukDeviceHandle *>(
      reinterpret_cast<Rhandle *>(qureg.rhandle)->env.ehandle);
}

static const int MAX_TASK_SIZE = (1LL << 16) / sizeof(Qtask);
__constant__ Qtask task_device[MAX_TASK_SIZE];

static __forceinline__ __device__ __host__ int
extractBit(const int locationOfBitFromRight,
           const long long int theEncodedNumber) {
  return (theEncodedNumber >> locationOfBitFromRight) & 1;
}

/** Copy from QuEST_cpu_distributed.c (may be adjusted a little) begin */

static int halfMatrixBlockFitsInChunk(long long int chunkSize,
                                      int targetQubit) {
  const long long int sizeHalfBlock = 1LL << (targetQubit);
  return chunkSize > sizeHalfBlock;
}

static int chunkIsUpper(int chunkId, long long int chunkSize, int targetQubit) {
  long long int sizeHalfBlock = 1LL << (targetQubit);
  long long int sizeBlock = sizeHalfBlock * 2;
  long long int posInBlock = (chunkId * chunkSize) % sizeBlock;
  return posInBlock < sizeHalfBlock;
}

static int getChunkPairId(int chunkIsUpper, int chunkId,
                          long long int chunkSize, int targetQubit) {
  long long int sizeHalfBlock = 1LL << (targetQubit);
  int chunksPerHalfBlock = sizeHalfBlock / chunkSize;
  if (chunkIsUpper) {
    return chunkId + chunksPerHalfBlock;
  } else {
    return chunkId - chunksPerHalfBlock;
  }
}

// declared these two function here to use them in exchangeStateVectors before
// defining

static int getChunkIdFromIndex(Qureg qureg, long long int index) {
  return index / qureg.numAmpsPerChunk; // this is numAmpsPerChunk
}

static void exchangeStateVectorsPart(ComplexArray deviceStateVec,
                                     ComplexArray devicePairStateVec,
                                     long long int numTasks, int pairRank) {
  std::vector<MPI_Request> request(
      (numTasks * sizeof(qreal) + MPI_MAX_AMPS_IN_MSG - 1) / MPI_MAX_AMPS_IN_MSG
      << 2); // 必须一开始就分配足够的Request，否则会重排导致指针失效！
  for (long long offset = 0, TAG = 0; offset < numTasks;
       offset += MPI_MAX_AMPS_IN_MSG, TAG += 2) {
    long long int maxMessageCount = MPI_MAX_AMPS_IN_MSG;
    if (maxMessageCount > numTasks - offset)
      maxMessageCount = numTasks - offset;
    MPI_Isend(&deviceStateVec.real[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG, MPI_COMM_WORLD, &request[TAG << 1 | 0]);
    MPI_Irecv(&devicePairStateVec.real[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG, MPI_COMM_WORLD, &request[TAG << 1 | 1]);
    MPI_Isend(&deviceStateVec.imag[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG + 1, MPI_COMM_WORLD, &request[TAG + 1 << 1 | 0]);
    MPI_Irecv(&devicePairStateVec.imag[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG + 1, MPI_COMM_WORLD, &request[TAG + 1 << 1 | 1]);
  }
  for (auto &r : request)
    MPI_Wait(&r, MPI_STATUSES_IGNORE);
}

template <int BLOCK_DIM_X>
static __global__
__launch_bounds__(BLOCK_DIM_X) void statevec_findAllProbabilityOfZeroKernel(
    const long long int offset, const long long int numTasks,
    const qreal *const stateVecReal, const qreal *const stateVecImag,
    qreal *const reducedArray, int PROBABILITY_PRE_CALCULATION) {
  const long long int measureQubit = threadIdx.x;
  qreal ans = 0;
  __shared__ qreal shmem[2][BLOCK_DIM_X];
  auto buffer = shmem[0], buffer1 = shmem[1];
  for (auto index = (long long int)blockIdx.x * BLOCK_DIM_X + measureQubit,
            stride = (long long int)gridDim.x * BLOCK_DIM_X;
       index < numTasks; index += stride) {
    do {
      auto t = make_cuDoubleComplex(stateVecReal[index], stateVecImag[index]);
      buffer[measureQubit] =
          fma(cuCreal(t), cuCreal(t), cuCimag(t) * cuCimag(t));
      __syncthreads();
    } while (0);
    if (measureQubit < PROBABILITY_PRE_CALCULATION)
#pragma unroll
      for (int j = 0; j < BLOCK_DIM_X; ++j)
        if (!extractBit(measureQubit, offset + index - measureQubit + j))
          ans += buffer[j];
    auto buffer2 = buffer1;
    buffer1 = buffer;
    buffer = buffer2;
  }
  if (measureQubit < PROBABILITY_PRE_CALCULATION)
    reducedArray[blockIdx.x * PROBABILITY_PRE_CALCULATION + measureQubit] = ans;
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_multiControlledPhaseShiftKernel(
    Qureg qureg, Qtask task) {
  auto stateVecSize = qureg.numAmpsPerChunk;
  auto index = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= stateVecSize)
    return;
  auto offset = qureg.numAmpsPerChunk * qureg.chunkId;
  auto mask = task.args.statevec_multiControlledPhaseShift.mask;
  auto ctrlFlipMask = task.args.statevec_multiControlledPhaseShift.ctrlFlipMask;
  if (mask == (mask & ((index + offset) ^ ctrlFlipMask))) {

    auto stateVecReal = qureg.deviceStateVec.real;
    auto stateVecImag = qureg.deviceStateVec.imag;

    auto stateLo =
        cuCmul(make_cuDoubleComplex(stateVecReal[index], stateVecImag[index]),
               task.args.statevec_multiControlledPhaseShift.term);
    stateVecReal[index] = cuCreal(stateLo);
    stateVecImag[index] = cuCimag(stateLo);
  }
}

static void statevec_multiControlledPhaseShift_doCompute(Qureg qureg,
                                                         Qtask task) {
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  CUDABlocks = ceil((qreal)qureg.numAmpsPerChunk / threadsPerCUDABlock);
  statevec_multiControlledPhaseShiftKernel<<<
      CUDABlocks, threadsPerCUDABlock, 0,
      get_wukDeviceHandle(qureg)->stream()>>>(qureg, task);
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_multiControlledPhaseShift_combine_Kernel(
    Qureg qureg, int task_n) {
  auto stateVecSize = qureg.numAmpsPerChunk;
  auto index = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= stateVecSize)
    return;
  auto offset = qureg.numAmpsPerChunk * qureg.chunkId;

  qreal *stateVecReal = qureg.deviceStateVec.real;
  qreal *stateVecImag = qureg.deviceStateVec.imag;
  auto stateLo = make_cuDoubleComplex(stateVecReal[index], stateVecImag[index]);
  for (int j = 0; j < task_n; ++j) {
    auto mask = task_device[j].args.statevec_multiControlledPhaseShift.mask;
    auto ctrlFlipMask =
        task_device[j].args.statevec_multiControlledPhaseShift.ctrlFlipMask;
    if (mask == (mask & ((index + offset) ^ ctrlFlipMask)))
      stateLo = cuCmul(
          stateLo, task_device[j].args.statevec_multiControlledPhaseShift.term);
  }
  stateVecReal[index] = cuCreal(stateLo);
  stateVecImag[index] = cuCimag(stateLo);
}

static void statevec_multiControlledPhaseShift_combine_doCompute(
    Qureg qureg, const int task_n, const Qtask *task_host) {
  if (task_n < 2) {
    for (int j = 0; j < task_n; ++j)
      statevec_multiControlledPhaseShift_doCompute(qureg, task_host[j]);
    return;
  }
  if (task_n > MAX_TASK_SIZE) {
    statevec_multiControlledPhaseShift_combine_doCompute(qureg, MAX_TASK_SIZE,
                                                         task_host);
    statevec_multiControlledPhaseShift_combine_doCompute(
        qureg, task_n - MAX_TASK_SIZE, task_host + MAX_TASK_SIZE);
    return;
  }
  cudaMemcpyToSymbolAsync(task_device, task_host, task_n * sizeof(Qtask), 0,
                          cudaMemcpyHostToDevice,
                          get_wukDeviceHandle(qureg)->stream());
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  CUDABlocks = ceil((qreal)qureg.numAmpsPerChunk / threadsPerCUDABlock);
  statevec_multiControlledPhaseShift_combine_Kernel<<<
      CUDABlocks, threadsPerCUDABlock, 0,
      get_wukDeviceHandle(qureg)->stream()>>>(qureg, task_n);
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_multiControlledUnitaryLocalKernel(
    Qureg qureg, Qtask task) {
  auto numTasks = qureg.numAmpsPerChunk >> 1;
  auto thisTask = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (thisTask >= numTasks)
    return;

  auto offset = qureg.numAmpsPerChunk * qureg.chunkId;

  auto sizeHalfBlock = 1LL << task.args.statevec_multiControlledUnitary
                                  .targetQubit; // size of blocks halved
  auto sizeBlock = 2LL * sizeHalfBlock;         // size of blocks

  // ---------------------------------------------------------------- //
  //            rotate                                                //
  // ---------------------------------------------------------------- //

  //! fix -- no necessary for GPU version
  qreal *stateVecReal = qureg.deviceStateVec.real;
  qreal *stateVecImag = qureg.deviceStateVec.imag;

  auto thisBlock =
      thisTask >> task.args.statevec_multiControlledUnitary.targetQubit;
  auto indexUp = thisBlock * sizeBlock + (thisTask & (sizeHalfBlock - 1));
  auto indexLo = indexUp + sizeHalfBlock;
  auto mask = task.args.statevec_multiControlledUnitary.mask;
  if (mask == (mask & (indexUp + offset))) {
    // store current state vector values in temp variables
    auto stateUp =
             make_cuDoubleComplex(stateVecReal[indexUp], stateVecImag[indexUp]),
         stateLo =
             make_cuDoubleComplex(stateVecReal[indexLo], stateVecImag[indexLo]);

    // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
    // state[indexLo] = u10 * state[indexUp] + u11 * state[indexLo]
    auto stateUp1 =
             cuCfma(task.args.statevec_multiControlledUnitary.u[0][0], stateUp,
                    cuCmul(task.args.statevec_multiControlledUnitary.u[0][1],
                           stateLo)),
         stateLo1 =
             cuCfma(task.args.statevec_multiControlledUnitary.u[1][0], stateUp,
                    cuCmul(task.args.statevec_multiControlledUnitary.u[1][1],
                           stateLo));

    stateVecReal[indexUp] = cuCreal(stateUp1);
    stateVecImag[indexUp] = cuCimag(stateUp1);

    stateVecReal[indexLo] = cuCreal(stateLo1);
    stateVecImag[indexLo] = cuCimag(stateLo1);
  }
}

template <int rankIsUpper>
static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_multiControlledUnitaryDistributedKernel(
    Qureg qureg, Qtask task, ComplexArray stateVecUp, ComplexArray stateVecLo,
    long long int offset, long long int numTasks) {
  auto thisTask = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (thisTask >= numTasks)
    return;
  auto mask = task.args.statevec_multiControlledUnitary.mask;
  if (mask == (mask & (thisTask + offset))) {

    auto stateVecRealUp = stateVecUp.real, stateVecImagUp = stateVecUp.imag,
         stateVecRealLo = stateVecLo.real, stateVecImagLo = stateVecLo.imag;
    auto stateUp = make_cuDoubleComplex(stateVecRealUp[thisTask],
                                        stateVecImagUp[thisTask]),
         stateLo = make_cuDoubleComplex(stateVecRealLo[thisTask],
                                        stateVecImagLo[thisTask]);

    // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
    // state[indexLo] = u10 * state[indexUp] + u11 * state[indexLo]
    auto stateUp1 =
             cuCfma(task.args.statevec_multiControlledUnitary.u[0][0], stateUp,
                    cuCmul(task.args.statevec_multiControlledUnitary.u[0][1],
                           stateLo)),
         stateLo1 =
             cuCfma(task.args.statevec_multiControlledUnitary.u[1][0], stateUp,
                    cuCmul(task.args.statevec_multiControlledUnitary.u[1][1],
                           stateLo));

    if (rankIsUpper) {
      stateVecRealUp[thisTask] = cuCreal(stateUp1);
      stateVecImagUp[thisTask] = cuCimag(stateUp1);
    } else {
      stateVecRealLo[thisTask] = cuCreal(stateLo1);
      stateVecImagLo[thisTask] = cuCimag(stateLo1);
    }
  }
}

static void statevec_multiControlledUnitary_doCompute(Qureg qureg, Qtask task) {
  const int threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  int CUDABlocks;

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at
  // most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(
      qureg.numAmpsPerChunk,
      task.args.statevec_multiControlledUnitary.targetQubit);

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly) {
    // all values required to update state vector lie in this rank
    CUDABlocks =
        ceil((qreal)(qureg.numAmpsPerChunk >> 1) / threadsPerCUDABlock);
    statevec_multiControlledUnitaryLocalKernel<<<
        CUDABlocks, threadsPerCUDABlock, 0,
        get_wukDeviceHandle(qureg)->stream()>>>(qureg, task);
  } else {
    // need to get corresponding chunk of state vector from other rank
    CUDABlocks = ceil((qreal)qureg.numAmpsPerChunk / threadsPerCUDABlock);
    rankIsUpper =
        chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk,
                     task.args.statevec_multiControlledUnitary.targetQubit);
    pairRank =
        getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk,
                       task.args.statevec_multiControlledUnitary.targetQubit);
    // printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
    // get corresponding values from my pair
    // exchangeStateVectors(qureg, pairRank);
    get_wukDeviceHandle(qureg)->sync();
    get_wukDeviceHandle(qureg)->fork();
    get_wukDeviceHandle(qureg)->device_store();
    std::vector<ComplexArray> devicePairStateVecBuffer(
        get_wukDeviceHandle(qureg)->get_num_active_streams() + 1);
    for (auto &it : devicePairStateVecBuffer) {
      it.real = get_wukDeviceHandle(qureg)->device_malloc<qreal>(BUFFER_AMP);
      it.imag = get_wukDeviceHandle(qureg)->device_malloc<qreal>(BUFFER_AMP);
    }
    for (long long int poffset = 0; poffset < qureg.numAmpsPerChunk;
         poffset += BUFFER_AMP) {
      long long int numTasks = BUFFER_AMP;
      if (numTasks > qureg.numAmpsPerChunk - poffset)
        numTasks = qureg.numAmpsPerChunk - poffset;
      auto deviceStateVec = qureg.deviceStateVec;
      deviceStateVec.real += poffset;
      deviceStateVec.imag += poffset;
      exchangeStateVectorsPart(deviceStateVec, devicePairStateVecBuffer.back(),
                               numTasks, pairRank);
      get_wukDeviceHandle(qureg)->revolve();
      cudaStreamSynchronize(get_wukDeviceHandle(qureg)->stream());
      // this rank's values are either in the upper of lower half of the block.
      // send values to controlledCompactUnitaryDistributed in the correct order
      if (rankIsUpper) {
        statevec_multiControlledUnitaryDistributedKernel<1>
            <<<CUDABlocks, threadsPerCUDABlock, 0,
               get_wukDeviceHandle(qureg)->stream()>>>(
                qureg, task,
                deviceStateVec,                  // upper
                devicePairStateVecBuffer.back(), // lower
                qureg.numAmpsPerChunk * qureg.chunkId + poffset, numTasks);
      } else {
        statevec_multiControlledUnitaryDistributedKernel<0>
            <<<CUDABlocks, threadsPerCUDABlock, 0,
               get_wukDeviceHandle(qureg)->stream()>>>(
                qureg, task,
                devicePairStateVecBuffer.back(), // upper
                deviceStateVec,                  // lower
                qureg.numAmpsPerChunk * qureg.chunkId + poffset, numTasks);
      }
      std::swap(devicePairStateVecBuffer.back(),
                devicePairStateVecBuffer[get_wukDeviceHandle(qureg)
                                             ->get_current_stream_index()]);
    }
    get_wukDeviceHandle(qureg)->join();
    get_wukDeviceHandle(qureg)->sync();
    get_wukDeviceHandle(qureg)->device_restore();
  }
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_multiControlledUnitary_combine_LocalKernel(
    Qureg qureg, const int task_n, const int targetQubit) {
  auto numTasks = qureg.numAmpsPerChunk >> 1;
  auto thisTask = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (thisTask >= numTasks)
    return;

  auto offset = qureg.numAmpsPerChunk * qureg.chunkId;

  auto sizeHalfBlock = 1LL << targetQubit; // size of blocks halved
  auto thisBlock = thisTask >> targetQubit;
  auto indexUp =
      (thisBlock * sizeHalfBlock << 1) + (thisTask & (sizeHalfBlock - 1));
  auto indexLo = indexUp + sizeHalfBlock;

  auto stateUp = make_cuDoubleComplex(qureg.deviceStateVec.real[indexUp],
                                      qureg.deviceStateVec.imag[indexUp]),
       stateLo = make_cuDoubleComplex(qureg.deviceStateVec.real[indexLo],
                                      qureg.deviceStateVec.imag[indexLo]);
  for (int j = 0; j < task_n; ++j) {
    auto mask = task_device[j].args.statevec_multiControlledUnitary.mask;
    if (mask == (mask & (indexUp + offset))) {
      auto stateUp1 = cuCfma(
               task_device[j].args.statevec_multiControlledUnitary.u[0][0],
               stateUp,
               cuCmul(
                   task_device[j].args.statevec_multiControlledUnitary.u[0][1],
                   stateLo)),
           stateLo1 = cuCfma(
               task_device[j].args.statevec_multiControlledUnitary.u[1][0],
               stateUp,
               cuCmul(
                   task_device[j].args.statevec_multiControlledUnitary.u[1][1],
                   stateLo));
      stateUp = stateUp1;
      stateLo = stateLo1;
    }
  }
  qureg.deviceStateVec.real[indexUp] = cuCreal(stateUp);
  qureg.deviceStateVec.imag[indexUp] = cuCimag(stateUp);

  qureg.deviceStateVec.real[indexLo] = cuCreal(stateLo);
  qureg.deviceStateVec.imag[indexLo] = cuCimag(stateLo);
}

template <int rankIsUpper>
static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_multiControlledUnitary_combine_DistributedKernel(
    Qureg qureg, const int task_n, const int targetQubit,
    ComplexArray stateVecUp, ComplexArray stateVecLo, long long int offset,
    long long int numTasks) {
  auto thisTask = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (thisTask >= numTasks)
    return;

  // store current state vector values in temp variables
  auto stateUp = make_cuDoubleComplex(stateVecUp.real[thisTask],
                                      stateVecUp.imag[thisTask]),
       stateLo = make_cuDoubleComplex(stateVecLo.real[thisTask],
                                      stateVecLo.imag[thisTask]);
  for (int j = 0; j < task_n; ++j) {
    auto mask = task_device[j].args.statevec_multiControlledUnitary.mask;
    if (mask == (mask & (thisTask + offset))) {
      auto stateUp1 = cuCfma(
               task_device[j].args.statevec_multiControlledUnitary.u[0][0],
               stateUp,
               cuCmul(
                   task_device[j].args.statevec_multiControlledUnitary.u[0][1],
                   stateLo)),
           stateLo1 = cuCfma(
               task_device[j].args.statevec_multiControlledUnitary.u[1][0],
               stateUp,
               cuCmul(
                   task_device[j].args.statevec_multiControlledUnitary.u[1][1],
                   stateLo));
      stateUp = stateUp1;
      stateLo = stateLo1;
    }
  }
  // store current state vector values in temp variables
  if (rankIsUpper) {
    stateVecUp.real[thisTask] = cuCreal(stateUp);
    stateVecUp.imag[thisTask] = cuCimag(stateUp);
  } else {
    stateVecLo.real[thisTask] = cuCreal(stateLo);
    stateVecLo.imag[thisTask] = cuCimag(stateLo);
  }
}

static void
statevec_multiControlledUnitary_combine_doCompute(Qureg qureg, const int task_n,
                                                  const Qtask *task_host) {
  if (task_n < 2) {
    for (int j = 0; j < task_n; ++j)
      statevec_multiControlledUnitary_doCompute(qureg, task_host[j]);
    return;
  }
  if (task_n > MAX_TASK_SIZE) {
    statevec_multiControlledUnitary_combine_doCompute(qureg, MAX_TASK_SIZE,
                                                      task_host);
    statevec_multiControlledUnitary_combine_doCompute(
        qureg, task_n - MAX_TASK_SIZE, task_host + MAX_TASK_SIZE);
    return;
  }
  cudaMemcpyToSymbolAsync(task_device, task_host, task_n * sizeof(Qtask), 0,
                          cudaMemcpyHostToDevice,
                          get_wukDeviceHandle(qureg)->stream());
  auto targetQubit =
      task_host->args.statevec_multiControlledUnitary.targetQubit;
  const int threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  int CUDABlocks;

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at
  // most half a block fits on one rank
  int useLocalDataOnly =
      halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly) {
    // all values required to update state vector lie in this rank
    CUDABlocks =
        ceil((qreal)(qureg.numAmpsPerChunk >> 1) / threadsPerCUDABlock);
    statevec_multiControlledUnitary_combine_LocalKernel<<<
        CUDABlocks, threadsPerCUDABlock, 0,
        get_wukDeviceHandle(qureg)->stream()>>>(qureg, task_n, targetQubit);
  } else {
    // need to get corresponding chunk of state vector from other rank
    CUDABlocks = ceil((qreal)qureg.numAmpsPerChunk / threadsPerCUDABlock);
    rankIsUpper =
        chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk,
                              targetQubit);
    // printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
    // get corresponding values from my pair
    // exchangeStateVectors(qureg, pairRank);
    get_wukDeviceHandle(qureg)->sync();
    get_wukDeviceHandle(qureg)->fork();
    get_wukDeviceHandle(qureg)->device_store();
    std::vector<ComplexArray> devicePairStateVecBuffer(
        get_wukDeviceHandle(qureg)->get_num_active_streams() + 1);
    for (auto &it : devicePairStateVecBuffer) {
      it.real = get_wukDeviceHandle(qureg)->device_malloc<qreal>(BUFFER_AMP);
      it.imag = get_wukDeviceHandle(qureg)->device_malloc<qreal>(BUFFER_AMP);
    }
    for (long long int poffset = 0; poffset < qureg.numAmpsPerChunk;
         poffset += BUFFER_AMP) {
      long long int numTasks = BUFFER_AMP;
      if (numTasks > qureg.numAmpsPerChunk - poffset)
        numTasks = qureg.numAmpsPerChunk - poffset;
      auto deviceStateVec = qureg.deviceStateVec;
      deviceStateVec.real += poffset;
      deviceStateVec.imag += poffset;
      exchangeStateVectorsPart(deviceStateVec, devicePairStateVecBuffer.back(),
                               numTasks, pairRank);
      get_wukDeviceHandle(qureg)->revolve();
      cudaStreamSynchronize(get_wukDeviceHandle(qureg)->stream());
      // this rank's values are either in the upper of lower half of the block.
      // send values to controlledCompactUnitaryDistributed in the correct order
      if (rankIsUpper) {
        statevec_multiControlledUnitary_combine_DistributedKernel<1>
            <<<CUDABlocks, threadsPerCUDABlock, 0,
               get_wukDeviceHandle(qureg)->stream()>>>(
                qureg, task_n, targetQubit,
                deviceStateVec,                  // upper
                devicePairStateVecBuffer.back(), // lower
                qureg.numAmpsPerChunk * qureg.chunkId + poffset, numTasks);
      } else {
        statevec_multiControlledUnitary_combine_DistributedKernel<0>
            <<<CUDABlocks, threadsPerCUDABlock, 0,
               get_wukDeviceHandle(qureg)->stream()>>>(
                qureg, task_n, targetQubit,
                devicePairStateVecBuffer.back(), // upper
                deviceStateVec,                  // lower
                qureg.numAmpsPerChunk * qureg.chunkId + poffset, numTasks);
      }
      std::swap(devicePairStateVecBuffer.back(),
                devicePairStateVecBuffer[get_wukDeviceHandle(qureg)
                                             ->get_current_stream_index()]);
    }
    get_wukDeviceHandle(qureg)->join();
    get_wukDeviceHandle(qureg)->sync();
    get_wukDeviceHandle(qureg)->device_restore();
  }
}

template <int BLOCK_DIM_X, int BLOCK_TASK>
static __global__
__launch_bounds__(BLOCK_DIM_X) void statevec_multiControlledUnitary_mixcombine_LocalKernel(
    Qureg qureg, qreal *stateVecReal, qreal *stateVecImag, const int task_n) {
  // ----- temp variables
  auto idx = threadIdx.x;
  do {
    auto numTasks = qureg.numAmpsPerChunk / (BLOCK_TASK / BLOCK_DIM_X);

    const long long int thisTask =
        (long long int)blockIdx.x * BLOCK_DIM_X + idx;
    // task based approach for expose loop with small granularity
    if (thisTask >= numTasks)
      return;
  } while (0);

  auto offset = qureg.numAmpsPerChunk * qureg.chunkId;
  extern __shared__ cuDoubleComplex state[];
  auto poffset = BLOCK_TASK * blockIdx.x;
#pragma unroll
  for (auto i = 0; i < BLOCK_TASK; i += BLOCK_DIM_X)
    state[i + idx] = make_cuDoubleComplex(stateVecReal[poffset + i + idx],
                                          stateVecImag[poffset + i + idx]);
  for (int j = 0; j < task_n; ++j) {
    auto targetQubit =
        task_device[j].args.statevec_multiControlledUnitary.targetQubit;
    long long int mask =
        task_device[j].args.statevec_multiControlledUnitary.mask;
    auto sizeHalfBlock = 1LL << targetQubit; // size of blocks halved
    auto sizeBlock = 2LL * sizeHalfBlock;    // size of blocks
    __syncthreads();
#pragma unroll
    for (auto i = 0; i < BLOCK_TASK / 2 / BLOCK_DIM_X; ++i) {
      auto myTask = i * BLOCK_DIM_X + idx;
      auto thisBlock = myTask >> targetQubit;
      auto indexUp = thisBlock * sizeBlock + (myTask & (sizeHalfBlock - 1));
      auto indexLo = indexUp + sizeHalfBlock;
      if (mask == (mask & (offset + poffset + indexUp))) {
        auto stateUp = state[indexUp];
        auto stateLo = state[indexLo];
        auto
            stateUp1 = cuCfma(
                task_device[j].args.statevec_multiControlledUnitary.u[0][0],
                stateUp,
                cuCmul(
                    task_device[j].args.statevec_multiControlledUnitary.u[0][1],
                    stateLo)),
            stateLo1 = cuCfma(
                task_device[j].args.statevec_multiControlledUnitary.u[1][0],
                stateUp,
                cuCmul(
                    task_device[j].args.statevec_multiControlledUnitary.u[1][1],
                    stateLo));
        state[indexUp] = stateUp1;
        state[indexLo] = stateLo1;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < BLOCK_TASK; i += BLOCK_DIM_X) {
    stateVecReal[poffset + i + idx] = cuCreal(state[i + idx]);
    stateVecImag[poffset + i + idx] = cuCimag(state[i + idx]);
  }
}

template <int CBIT>
static void statevec_multiControlledUnitary_mixcombine_doCompute(
    Qureg qureg, const int task_n, const Qtask *task_host) {
  if (task_n < 2) {
    for (int j = 0; j < task_n; ++j)
      statevec_multiControlledUnitary_doCompute(qureg, task_host[j]);
    return;
  }
  if (task_n > MAX_TASK_SIZE) {
    statevec_multiControlledUnitary_mixcombine_doCompute<CBIT>(
        qureg, MAX_TASK_SIZE, task_host);
    statevec_multiControlledUnitary_mixcombine_doCompute<CBIT>(
        qureg, task_n - MAX_TASK_SIZE, task_host + MAX_TASK_SIZE);
    return;
  }
  cudaMemcpyToSymbolAsync(task_device, task_host, task_n * sizeof(Qtask), 0,
                          cudaMemcpyHostToDevice,
                          get_wukDeviceHandle(qureg)->stream());
  const int threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  const int BLOCK_TASK = 2LL << CBIT;
  int CUDABlocks =
      ceil((qreal)(qureg.numAmpsPerChunk / (BLOCK_TASK / threadsPerCUDABlock)) /
           threadsPerCUDABlock);
  static_assert(BLOCK_TASK / threadsPerCUDABlock >= 2);
  static_assert(BLOCK_TASK % threadsPerCUDABlock == 0);
  cudaFuncSetAttribute(statevec_multiControlledUnitary_mixcombine_LocalKernel<
                           threadsPerCUDABlock, BLOCK_TASK>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       BLOCK_TASK * sizeof(cuDoubleComplex)); // F**K
  statevec_multiControlledUnitary_mixcombine_LocalKernel<threadsPerCUDABlock,
                                                         BLOCK_TASK>
      <<<CUDABlocks, threadsPerCUDABlock, BLOCK_TASK * sizeof(cuDoubleComplex),
         get_wukDeviceHandle(qureg)->stream()>>>(
          qureg, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, task_n);
}

static bool doCompute(Qureg qureg) {
  auto &tbd = reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd;
  if (tbd.empty())
    return 0;

  for (int i = 0; i < (int)tbd.size(); ++i) {
    switch (tbd[i].func) {
    case Qfunc_statevec_multiControlledUnitary: {
      auto targetQubit =
          tbd[i].args.statevec_multiControlledUnitary.targetQubit;
      int task_n = 1;
      if (targetQubit <= COMBINE_BIT) {
        while (
            i + task_n < (int)tbd.size() &&
            tbd[i + task_n].func == Qfunc_statevec_multiControlledUnitary &&
            tbd[i + task_n].args.statevec_multiControlledUnitary.targetQubit <=
                COMBINE_BIT)
          ++task_n;
        statevec_multiControlledUnitary_mixcombine_doCompute<COMBINE_BIT>(
            qureg, task_n, tbd.data() + i);
      } else {
        while (
            i + task_n < (int)tbd.size() &&
            tbd[i + task_n].func == Qfunc_statevec_multiControlledUnitary &&
            tbd[i + task_n].args.statevec_multiControlledUnitary.targetQubit ==
                targetQubit)
          ++task_n;
        statevec_multiControlledUnitary_combine_doCompute(qureg, task_n,
                                                          tbd.data() + i);
      }
      i += task_n - 1;
      break;
    }
    case Qfunc_statevec_multiControlledPhaseShift: {
      int task_n = 1;
      while (i + task_n < (int)tbd.size() &&
             tbd[i + task_n].func == Qfunc_statevec_multiControlledPhaseShift)
        ++task_n;
      statevec_multiControlledPhaseShift_combine_doCompute(qureg, task_n,
                                                           tbd.data() + i);
      i += task_n - 1;
      break;
    }
    default:
      printf("%d not been supported yet.\n", tbd[i].func);
      exit(EXIT_FAILURE);
    };
  }
  get_wukDeviceHandle(qureg)->sync();
  tbd.clear();
  return 1;
}

static void copyStateToGPU(Qureg qureg) {
  cudaMemcpyAsync(qureg.deviceStateVec.real, qureg.stateVec.real,
                  qureg.numAmpsPerChunk * sizeof(*(qureg.deviceStateVec.real)),
                  cudaMemcpyHostToDevice, get_wukDeviceHandle(qureg)->stream());
  cudaMemcpyAsync(qureg.deviceStateVec.imag, qureg.stateVec.imag,
                  qureg.numAmpsPerChunk * sizeof(*(qureg.deviceStateVec.imag)),
                  cudaMemcpyHostToDevice, get_wukDeviceHandle(qureg)->stream());
}

static void copyStateFromGPU(Qureg qureg) {
  cudaMemcpyAsync(qureg.stateVec.real, qureg.deviceStateVec.real,
                  qureg.numAmpsPerChunk * sizeof(*(qureg.deviceStateVec.real)),
                  cudaMemcpyDeviceToHost, get_wukDeviceHandle(qureg)->stream());
  cudaMemcpyAsync(qureg.stateVec.imag, qureg.deviceStateVec.imag,
                  qureg.numAmpsPerChunk * sizeof(*(qureg.deviceStateVec.imag)),
                  cudaMemcpyDeviceToHost, get_wukDeviceHandle(qureg)->stream());
}

#if defined(__cplusplus)
extern "C" {
#endif
void seedQuESTDefault() {
  // init MT random number generator with three keys -- time and pid
  // for the MPI version, it is ok that all procs will get the same seed as
  // random numbers will only be used by the master process

  unsigned long int key[2];
  getQuESTDefaultSeedKey(key);
  // this seed will be used to generate the same random number on all procs,
  // therefore we want to make sure all procs receive the same key
  MPI_Bcast(key, 2, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  init_by_array(key, 2);
}

void statevec_createQureg(Qureg *q, int numQubits, QuESTEnv env) {
  // allocate CPU memory
  Qureg qureg;

  long long int numAmps = 1L << numQubits;
  long long int numAmpsPerRank = numAmps / env.numRanks;

  qureg.numQubitsInStateVec = numQubits;
  qureg.numAmpsPerChunk = numAmpsPerRank;
  qureg.numAmpsTotal = numAmps;
  qureg.chunkId = env.rank;
  qureg.numChunks = env.numRanks;
  qureg.isDensityMatrix = 0;

  qureg.rhandle = reinterpret_cast<wukDeviceHandle *>(env.ehandle)
                      ->device_malloc_pinned<Rhandle *>(1);
  reinterpret_cast<Rhandle *>(qureg.rhandle)->env = env;
  reinterpret_cast<Rhandle *>(qureg.rhandle)->prob =
      std::vector<qreal>(numQubits, -1);

  qureg.deviceStateVec.real =
      get_wukDeviceHandle(qureg)->device_malloc<qreal>(numAmpsPerRank);
  qureg.deviceStateVec.imag =
      get_wukDeviceHandle(qureg)->device_malloc<qreal>(numAmpsPerRank);
  qureg.stateVec.real =
      get_wukDeviceHandle(qureg)->device_malloc_pinned<qreal>(numAmpsPerRank);
  qureg.stateVec.imag =
      get_wukDeviceHandle(qureg)->device_malloc_pinned<qreal>(numAmpsPerRank);

  *q = qureg;
}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env) {}

void syncQuESTEnv(QuESTEnv env) {
  reinterpret_cast<wukDeviceHandle *>(env.ehandle)->sync();
  MPI_Barrier(MPI_COMM_WORLD);
}

int syncQuESTSuccess(int successCode) {
  int totalSuccess;
  MPI_Allreduce(&successCode, &totalSuccess, 1, MPI_INT, MPI_LAND,
                MPI_COMM_WORLD);
  return totalSuccess;
}

QuESTEnv createQuESTEnv(void) {

  QuESTEnv env;

  // init MPI environment
  do {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      MPI_Init(NULL, NULL);
    } else {
      printf(
          "ERROR: Trying to initialize QuESTEnv multiple times. Ignoring...\n");
    }
    // ensure env is initialised anyway, so the compiler is happy
    MPI_Comm_size(MPI_COMM_WORLD, &env.numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
  } while (0);
  do {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, env.rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = -1, gpuDeviceCount;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    cudaGetDeviceCount(&gpuDeviceCount);
    int device = env.rank;//(local_rank + 1 ) % gpuDeviceCount;
    env.ehandle = new wukDeviceHandle(device);
  } while (0);

  seedQuESTDefault();

#if EXCHANGE_WARMUP_BIT > 0
  do {
    reinterpret_cast<wukDeviceHandle *>(env.ehandle)->device_store();
    std::vector<ComplexArray> devicePairStateVecBuffer(2);
    for (auto &it : devicePairStateVecBuffer) {
      it.real = reinterpret_cast<wukDeviceHandle *>(env.ehandle)
                    ->device_malloc<qreal>(BUFFER_AMP);
      it.imag = reinterpret_cast<wukDeviceHandle *>(env.ehandle)
                    ->device_malloc<qreal>(BUFFER_AMP);
    }
    for (int i = env.numRanks - 1; i >= 0; --i) {
      exchangeStateVectorsPart(devicePairStateVecBuffer.front(),
                               devicePairStateVecBuffer.back(), BUFFER_AMP,
                               env.rank ^ i);
    }
    reinterpret_cast<wukDeviceHandle *>(env.ehandle)->device_restore();
  } while (0);
  do {
    reinterpret_cast<wukDeviceHandle *>(env.ehandle)->device_store();
    reinterpret_cast<wukDeviceHandle *>(env.ehandle)->device_store_pinned();
    const int N = EXCHANGE_WARMUP_BIT;
    Qureg q = createQureg(N, env);
    /* GHZ quantum circuit */
    hadamard(q, 0);
    for (int i = 0; i < N - 1; ++i)
      controlledNot(q, i, i + 1);
    /* end of GHZ circuit */

    /* QFT starts */
    for (int i = 0; i < N - 1; ++i) {
      for (int j = 0; j < i; ++j)
        controlledRotateZ(q, j, i, M_PI * pow(0.5, i - j));
      hadamard(q, i);
    }
    /* end of QFT circuit */
    for (int i = 0; i < N; ++i) {
      auto tmp = calcProbOfOutcome(q, i, 1);
    }
    destroyQureg(q, env);
    reinterpret_cast<wukDeviceHandle *>(env.ehandle)->device_restore_pinned();
    reinterpret_cast<wukDeviceHandle *>(env.ehandle)->device_restore();
  } while (0);
#endif

  syncQuESTEnv(env);
  return env;
}

void destroyQuESTEnv(QuESTEnv env) {
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
    delete reinterpret_cast<wukDeviceHandle *>(env.ehandle);
    MPI_Finalize();
  } else
    printf("ERROR: Trying to close QuESTEnv multiple times. Ignoring...\n");
}

void reportQuESTEnv(QuESTEnv env) {
  if (env.rank == 0) {
    printf("EXECUTION ENVIRONMENT:\n");
    printf("Running distributed (MPI) GPU version\n");
    printf("Number of ranks is %d\n", env.numRanks);
    printf("OpenMP disabled\n");
    printf("Precision: size of qreal is %ld bytes\n", sizeof(qreal));
  }
}

void getEnvironmentString(QuESTEnv env, Qureg qureg, char str[]) {
  sprintf(str, "%dqubits_GPU_%dranks", qureg.numQubitsInStateVec, env.numRanks);
}

/** Print the current state vector of probability amplitudes for a set of
qubits to standard out. For debugging purposes. Each rank should print output
serially. Only print output for systems <= 5 qubits
*/
void statevec_reportStateToScreen(Qureg qureg, QuESTEnv env, int reportRank) {
  if (qureg.numQubitsInStateVec <= 5) {
    doCompute(qureg);
    copyStateFromGPU(qureg);
    for (int rank = 0; rank < qureg.numChunks; rank++) {
      syncQuESTEnv(env);
      if (qureg.chunkId == rank) {
        if (reportRank) {
          printf("Reporting state from rank %d [\n", qureg.chunkId);
          // printf("\trank, index, real, imag\n");
          printf("real, imag\n");
        } else if (rank == 0) {
          printf("Reporting state [\n");
          printf("real, imag\n");
        }

        for (long long int index = 0; index < qureg.numAmpsPerChunk; index++) {
          printf(REAL_STRING_FORMAT ", " REAL_STRING_FORMAT "\n",
                 qureg.stateVec.real[index], qureg.stateVec.imag[index]);
        }
        if (reportRank || rank == qureg.numChunks - 1)
          printf("]\n");
      }
    }
  } else
    printf("Error: reportStateToScreen will not print output for systems of "
           "more than 5 qubits.\n");
}

/** works for both statevectors and density matrices */
void statevec_cloneQureg(Qureg targetQureg, Qureg copyQureg) {
  // copy copyQureg's GPU statevec to targetQureg's GPU statevec
  reinterpret_cast<Rhandle *>(targetQureg.rhandle)->tbd.clear();
  doCompute(copyQureg);
  cudaMemcpyAsync(
      targetQureg.deviceStateVec.real, copyQureg.deviceStateVec.real,
      targetQureg.numAmpsPerChunk * sizeof(*(targetQureg.deviceStateVec.real)),
      cudaMemcpyDeviceToDevice, get_wukDeviceHandle(copyQureg)->stream());
  cudaMemcpyAsync(
      targetQureg.deviceStateVec.imag, copyQureg.deviceStateVec.imag,
      targetQureg.numAmpsPerChunk * sizeof(*(targetQureg.deviceStateVec.imag)),
      cudaMemcpyDeviceToDevice, get_wukDeviceHandle(copyQureg)->stream());
  get_wukDeviceHandle(copyQureg)->sync();
}

/** Copy from QuEST_cpu_distributed.c end */

void statevec_setAmps(Qureg qureg, long long int startInd, qreal *reals,
                      qreal *imags, long long int numAmps) {
  /* this is actually distributed, since the user's code runs on every node */

  // local start/end indices of the given amplitudes, assuming they fit in
  // this chunk these may be negative or above qureg.numAmpsPerChunk
  long long int localStartInd =
      startInd - qureg.chunkId * qureg.numAmpsPerChunk;
  long long int localEndInd = localStartInd + numAmps; // exclusive

  // add this to a local index to get corresponding elem in reals & imags
  long long int offset = qureg.chunkId * qureg.numAmpsPerChunk - startInd;

  // restrict these indices to fit into this chunk
  if (localStartInd < 0)
    localStartInd = 0;
  if (localEndInd > qureg.numAmpsPerChunk)
    localEndInd = qureg.numAmpsPerChunk;
  // they may now be out of order = no iterations

  if (localStartInd > localEndInd)
    return;
  cudaMemcpyAsync(
      qureg.deviceStateVec.real + localStartInd, reals + offset + localStartInd,
      (localEndInd - localStartInd) * sizeof(*(qureg.deviceStateVec.real)),
      cudaMemcpyHostToDevice, get_wukDeviceHandle(qureg)->stream());
  cudaMemcpyAsync(
      qureg.deviceStateVec.imag + localStartInd, imags + offset + localStartInd,
      (localEndInd - localStartInd) * sizeof(*(qureg.deviceStateVec.real)),
      cudaMemcpyHostToDevice, get_wukDeviceHandle(qureg)->stream());
}

qreal statevec_getRealAmp(Qureg qureg, long long int index) {
  int targetChunkId = getChunkIdFromIndex(qureg, index);
  qreal el = 0;
  doCompute(qureg);
  if (qureg.chunkId == targetChunkId) {
    cudaMemcpyAsync(&el,
                    qureg.deviceStateVec.real +
                        (index - targetChunkId * qureg.numAmpsPerChunk),
                    sizeof(*(qureg.deviceStateVec.real)),
                    cudaMemcpyDeviceToHost,
                    get_wukDeviceHandle(qureg)->stream());
    get_wukDeviceHandle(qureg)->sync();
  }
  MPI_Bcast(&el, 1, MPI_QuEST_REAL, targetChunkId, MPI_COMM_WORLD);
  return el;
}

qreal statevec_getImagAmp(Qureg qureg, long long int index) {
  int targetChunkId = getChunkIdFromIndex(qureg, index);
  qreal el = 0;
  doCompute(qureg);
  if (qureg.chunkId == targetChunkId) {
    cudaMemcpyAsync(&el,
                    qureg.deviceStateVec.imag +
                        (index - targetChunkId * qureg.numAmpsPerChunk),
                    sizeof(*(qureg.deviceStateVec.imag)),
                    cudaMemcpyDeviceToHost,
                    get_wukDeviceHandle(qureg)->stream());
    get_wukDeviceHandle(qureg)->sync();
  }
  MPI_Bcast(&el, 1, MPI_QuEST_REAL, targetChunkId, MPI_COMM_WORLD);
  return el;
}

void statevec_initZeroState(Qureg qureg) {
  statevec_initClassicalState(qureg, 0);
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_initPlusStateKernel(
    long long chunkSize, long long stateVecSize, qreal normFactor,
    qreal *stateVecReal, qreal *stateVecImag) {
  auto index = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= chunkSize)
    return;
  stateVecReal[index] = normFactor;
  stateVecImag[index] = 0.0;
}

void statevec_initPlusState(Qureg qureg) {
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.clear();
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  auto stateVecSize = qureg.numAmpsPerChunk * qureg.numChunks;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk) / threadsPerCUDABlock);
  statevec_initPlusStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk, stateVecSize, 1.0 / sqrt((qreal)stateVecSize),
      qureg.deviceStateVec.real, qureg.deviceStateVec.imag);
}

void statevec_initClassicalState(Qureg qureg, long long int stateInd) {
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.clear();
  cudaMemsetAsync(qureg.deviceStateVec.real, 0,
                  qureg.numAmpsPerChunk * sizeof(*(qureg.deviceStateVec.real)),
                  get_wukDeviceHandle(qureg)->stream());
  cudaMemsetAsync(qureg.deviceStateVec.imag, 0,
                  qureg.numAmpsPerChunk * sizeof(*(qureg.deviceStateVec.imag)),
                  get_wukDeviceHandle(qureg)->stream());
  static qreal reals = 1.0, imags = 0.0;
  statevec_setAmps(qureg, stateInd, &reals, &imags, 1);
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_initDebugStateKernel(
    long long int stateVecSize, qreal *stateVecReal, qreal *stateVecImag,
    int indexOffset) {
  long long int index;

  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= stateVecSize)
    return;

  stateVecReal[index] = ((index + indexOffset) * 2.0) / 10.0;
  stateVecImag[index] = ((index + indexOffset) * 2.0 + 1.0) / 10.0;
}

void statevec_initStateDebug(Qureg qureg) {
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.clear();
  long long int chunkSize;
  long long int indexOffset;
  int threadsPerCUDABlock, CUDABlocks;

  threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk) / threadsPerCUDABlock);

  // dimension of the state vector
  chunkSize = qureg.numAmpsPerChunk;

  indexOffset = chunkSize * qureg.chunkId;

  statevec_initDebugStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk, qureg.deviceStateVec.real,
      qureg.deviceStateVec.imag, indexOffset);
}

static __global__
__launch_bounds__(THREADS_PER_CUDA_BLOCK) void statevec_initStateOfSingleQubitKernel(
    Qureg qureg, qreal *stateVecReal, qreal *stateVecImag, int qubitId,
    int outcome) {
  auto index = (long long int)blockIdx.x * blockDim.x + threadIdx.x;

  // dimension of the state vector
  auto chunkSize = qureg.numAmpsPerChunk;
  if (index >= chunkSize)
    return;
  auto stateVecSize = chunkSize * qureg.numChunks;
  auto chunkId = qureg.chunkId;

  // initialise the state to |0000..0000>
  auto bit = extractBit(qubitId, index + chunkId * chunkSize);
  if (bit == outcome) {
    const qreal normFactor = rsqrt((qreal)stateVecSize / 2.0);
    stateVecReal[index] = normFactor;
    stateVecImag[index] = 0.0;
  } else {
    stateVecReal[index] = 0.0;
    stateVecImag[index] = 0.0;
  }
}

void statevec_initStateOfSingleQubit(Qureg *qureg, int qubitId, int outcome) {
  reinterpret_cast<Rhandle *>(qureg->rhandle)->tbd.clear();
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = THREADS_PER_CUDA_BLOCK;
  CUDABlocks = ceil((qreal)(qureg->numAmpsPerChunk) / threadsPerCUDABlock);
  statevec_initStateOfSingleQubitKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      *qureg, qureg->deviceStateVec.real, qureg->deviceStateVec.imag, qubitId,
      outcome);
}

// returns 1 if successful, else 0
int statevec_initStateFromSingleFile(Qureg *qureg, char filename[],
                                     QuESTEnv env) {
  reinterpret_cast<Rhandle *>(qureg->rhandle)->tbd.clear();
  long long int chunkSize, stateVecSize;
  long long int indexInChunk, totalIndex;
  chunkSize = qureg->numAmpsPerChunk;
  stateVecSize = chunkSize * qureg->numChunks;

  qreal *stateVecReal = qureg->stateVec.real;
  qreal *stateVecImag = qureg->stateVec.imag;

  FILE *fp;
  char line[200];

  fp = fopen(filename, "r");
  if (fp == NULL)
    return 0;

  indexInChunk = 0;
  totalIndex = 0;
  while (fgets(line, sizeof(char) * 200, fp) != NULL &&
         totalIndex < stateVecSize) {
    if (line[0] != '#') {
      int chunkId = totalIndex / chunkSize;
      if (chunkId == qureg->chunkId) {
#if QuEST_PREC == 1
        sscanf(line, "%f, %f", &(stateVecReal[indexInChunk]),
               &(stateVecImag[indexInChunk]));
#elif QuEST_PREC == 2
        sscanf(line, "%lf, %lf", &(stateVecReal[indexInChunk]),
               &(stateVecImag[indexInChunk]));
#elif QuEST_PREC == 4
        sscanf(line, "%lf, %lf", &(stateVecReal[indexInChunk]),
               &(stateVecImag[indexInChunk]));
#endif
        indexInChunk += 1;
      }
      totalIndex += 1;
    }
  }
  fclose(fp);
  copyStateToGPU(*qureg);

  // indicate success
  return 1;
}

int statevec_compareStates(Qureg mq1, Qureg mq2, qreal precision) {
  qreal diff;
  int chunkSize = mq1.numAmpsPerChunk;

  doCompute(mq1);
  doCompute(mq2);
  copyStateFromGPU(mq2);
  copyStateFromGPU(mq1);
  get_wukDeviceHandle(mq1)->sync();
  get_wukDeviceHandle(mq2)->sync();
  for (int i = 0; i < chunkSize; i++) {
    diff = mq1.stateVec.real[i] - mq2.stateVec.real[i];
    if (diff < 0)
      diff *= -1;
    if (diff > precision)
      return 0;
    diff = mq1.stateVec.imag[i] - mq2.stateVec.imag[i];
    if (diff < 0)
      diff *= -1;
    if (diff > precision)
      return 0;
  }
  return 1;
}

void statevec_compactUnitary(Qureg qureg, const int targetQubit, Complex alpha,
                             Complex beta) {
  statevec_controlledCompactUnitary(qureg, -1, targetQubit, alpha, beta);
}

void statevec_controlledCompactUnitary(Qureg qureg, const int controlQubit,
                                       const int targetQubit, Complex alpha,
                                       Complex beta) {
  ComplexMatrix2 u;
  u.r0c0 = alpha;
  u.r0c1.real = -beta.real;
  u.r0c1.imag = beta.imag;
  u.r1c0 = beta;
  u.r1c1.real = alpha.real;
  u.r1c1.imag = -alpha.imag;
  statevec_controlledUnitary(qureg, controlQubit, targetQubit, u);
}

void statevec_unitary(Qureg qureg, const int targetQubit, ComplexMatrix2 u) {
  statevec_controlledUnitary(qureg, -1, targetQubit, u);
}

void statevec_controlledUnitary(Qureg qureg, const int controlQubit,
                                const int targetQubit, ComplexMatrix2 u) {
  int controlQubits[] = {controlQubit};
  statevec_multiControlledUnitary(qureg, controlQubits, 1, targetQubit, u);
}

void statevec_multiControlledUnitary(Qureg qureg, int *controlQubits,
                                     int numControlQubits,
                                     const int targetQubit, ComplexMatrix2 u) {
  long long int mask = 0;
  for (int i = 0; i < numControlQubits; ++i)
    if (controlQubits[i] >= 0)
      mask |= 1LL << controlQubits[i];
  Qtask task;
  task.func = Qfunc_statevec_multiControlledUnitary;
  task.args.statevec_multiControlledUnitary.mask = mask;
  task.args.statevec_multiControlledUnitary.targetQubit = targetQubit;
  task.args.statevec_multiControlledUnitary.u[0][0] =
      make_cuDoubleComplex(u.r0c0.real, u.r0c0.imag);
  task.args.statevec_multiControlledUnitary.u[0][1] =
      make_cuDoubleComplex(u.r0c1.real, u.r0c1.imag);
  task.args.statevec_multiControlledUnitary.u[1][0] =
      make_cuDoubleComplex(u.r1c0.real, u.r1c0.imag);
  task.args.statevec_multiControlledUnitary.u[1][1] =
      make_cuDoubleComplex(u.r1c1.real, u.r1c1.imag);
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.push_back(task);
}

void statevec_pauliX(Qureg qureg, const int targetQubit) {
  ComplexMatrix2 u;
  u.r0c0.real = 0;
  u.r0c0.imag = 0;
  u.r0c1.real = 1;
  u.r0c1.imag = 0;
  u.r1c0 = u.r0c1;
  u.r1c1 = u.r0c0;
  statevec_unitary(qureg, targetQubit, u);
}

void statevec_pauliY(Qureg qureg, const int targetQubit) {
  statevec_controlledPauliY(qureg, -1, targetQubit);
}

void statevec_pauliYConj(Qureg qureg, const int targetQubit) {
  statevec_controlledPauliYConj(qureg, -1, targetQubit);
}

void statevec_controlledPauliY(Qureg qureg, const int controlQubit,
                               const int targetQubit) {
  qreal conjFac = 1;
  ComplexMatrix2 u;
  u.r0c0.real = 0;
  u.r0c0.imag = 0;
  u.r0c1.real = 0;
  u.r0c1.imag = -conjFac;
  u.r1c0.real = 0;
  u.r1c0.imag = conjFac;
  u.r1c1 = u.r0c0;
  statevec_controlledUnitary(qureg, controlQubit, targetQubit, u);
}

void statevec_controlledPauliYConj(Qureg qureg, const int controlQubit,
                                   const int targetQubit) {
  qreal conjFac = -1;
  ComplexMatrix2 u;
  u.r0c0.real = 0;
  u.r0c0.imag = 0;
  u.r0c1.real = 0;
  u.r0c1.imag = -conjFac;
  u.r1c0.real = 0;
  u.r1c0.imag = conjFac;
  u.r1c1 = u.r0c0;
  statevec_controlledUnitary(qureg, controlQubit, targetQubit, u);
}

static void statevec_phaseShiftByTerm_pushTask(Qureg qureg,
                                               const long long int mask,
                                               const long long int ctrlFlipMask,
                                               cuDoubleComplex term) {
  Qtask task;
  task.func = Qfunc_statevec_multiControlledPhaseShift;
  task.args.statevec_multiControlledPhaseShift.mask = mask;
  task.args.statevec_multiControlledPhaseShift.ctrlFlipMask = ctrlFlipMask;
  task.args.statevec_multiControlledPhaseShift.term = term;
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.push_back(task);
}
void statevec_phaseShiftByTerm(Qureg qureg, const int targetQubit,
                               Complex term) {
  statevec_phaseShiftByTerm_pushTask(
      qureg, 1LL << targetQubit, 0, make_cuDoubleComplex(term.real, term.imag));
}

void statevec_controlledPhaseShift(Qureg qureg, const int idQubit1,
                                   const int idQubit2, qreal angle) {
  int controlQubits[] = {idQubit1, idQubit2};
  statevec_multiControlledPhaseShift(qureg, controlQubits, 2, angle);
}

void statevec_multiControlledPhaseShift(Qureg qureg, int *controlQubits,
                                        int numControlQubits, qreal angle) {
  long long int mask = 0;
  for (int i = 0; i < numControlQubits; i++)
    if (controlQubits[i] >= 0)
      mask |= 1LL << controlQubits[i];
  statevec_phaseShiftByTerm_pushTask(
      qureg, mask, 0, make_cuDoubleComplex(cos(angle), sin(angle)));
}

void statevec_controlledPhaseFlip(Qureg qureg, const int idQubit1,
                                  const int idQubit2) {
  int controlQubits[] = {idQubit1, idQubit2};
  statevec_multiControlledPhaseFlip(qureg, controlQubits, 2);
}

void statevec_multiControlledPhaseFlip(Qureg qureg, int *controlQubits,
                                       int numControlQubits) {
  statevec_multiControlledPhaseShift(qureg, controlQubits, numControlQubits,
                                     M_PI * 0.5);
}

void statevec_hadamard(Qureg qureg, const int targetQubit) {
  ComplexMatrix2 u;
  qreal recRoot2 = 1.0 / sqrt((qreal)2);
  u.r0c0.real = recRoot2;
  u.r0c0.imag = 0;
  u.r0c1 = u.r0c0;
  u.r1c0.real = recRoot2;
  u.r1c0.imag = 0;
  u.r1c1.real = -recRoot2;
  u.r1c1.imag = 0;
  statevec_unitary(qureg, targetQubit, u);
}

void statevec_controlledNot(Qureg qureg, const int controlQubit,
                            const int targetQubit) {
  ComplexMatrix2 u;
  u.r0c0.real = 0;
  u.r0c0.imag = 0;
  u.r0c1.real = 1;
  u.r0c1.imag = 0;
  u.r1c0 = u.r0c1;
  u.r1c1 = u.r0c0;
  statevec_controlledUnitary(qureg, controlQubit, targetQubit, u);
}

qreal statevec_calcTotalProb(Qureg qureg) {
  doCompute(qureg);
  qreal ans = get_wukDeviceHandle(qureg)->dot(qureg.numAmpsPerChunk,
                                              qureg.deviceStateVec.real, 1,
                                              qureg.deviceStateVec.real, 1) +
              get_wukDeviceHandle(qureg)->dot(qureg.numAmpsPerChunk,
                                              qureg.deviceStateVec.imag, 1,
                                              qureg.deviceStateVec.imag, 1);
  MPI_Allreduce(MPI_IN_PLACE, &ans, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
  return ans;
}

qreal statevec_findProbabilityOfZero(Qureg qureg, const int measureQubit) {
  if (doCompute(qureg) ||
      reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.at(measureQubit) < 0) {
    get_wukDeviceHandle(qureg)->device_store();
    const long long int REDUCE_BUFFER_SIZE = 1LL << 15;
    auto firstLevelReduction =
        get_wukDeviceHandle(qureg)->device_malloc<qreal>(REDUCE_BUFFER_SIZE);
    const long long int numValuesToReduce = qureg.numAmpsPerChunk,
                        valuesPerCUDABlock = 256,
                        numCUDABlocks =
                            REDUCE_BUFFER_SIZE /
                            reinterpret_cast<Rhandle *>(qureg.rhandle)
                                ->prob.size();
    statevec_findAllProbabilityOfZeroKernel<valuesPerCUDABlock>
        <<<numCUDABlocks, valuesPerCUDABlock, 0,
           get_wukDeviceHandle(qureg)->stream()>>>(
            qureg.numAmpsPerChunk * qureg.chunkId, numValuesToReduce,
            qureg.deviceStateVec.real, qureg.deviceStateVec.imag,
            firstLevelReduction,
            reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.size());
    for (int measureQubit = 0;
         measureQubit <
         (int)reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.size();
         ++measureQubit)
      reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.at(measureQubit) =
          get_wukDeviceHandle(qureg)->asum(
              numCUDABlocks, firstLevelReduction + measureQubit,
              reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.size());
    get_wukDeviceHandle(qureg)->device_restore();
    MPI_Allreduce(MPI_IN_PLACE,
                  reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.data(),
                  reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.size(),
                  MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
  return reinterpret_cast<Rhandle *>(qureg.rhandle)->prob.at(measureQubit);
}

qreal statevec_calcProbOfOutcome(Qureg qureg, const int measureQubit,
                                 int outcome) {
  qreal totalStateProb = statevec_findProbabilityOfZero(qureg, measureQubit);
  if (outcome == 1)
    totalStateProb = 1.0 - totalStateProb;
  return totalStateProb;
}

/** Terrible code which unnecessarily individually computes and sums the real
 * and imaginary components of the inner product, so as to not have to worry
 * about keeping the sums separated during reduction. Truly disgusting,
 * probably doubles runtime, please fix.
 * @TODO could even do the kernel twice, storing real in bra.reduc and imag in
 * ket.reduc?
 */
Complex statevec_calcInnerProduct(Qureg bra, Qureg ket) {
  doCompute(bra);
  doCompute(ket);
  auto numAmps = bra.numAmpsPerChunk;
  qreal *braVecReal = bra.deviceStateVec.real;
  qreal *braVecImag = bra.deviceStateVec.imag;
  qreal *ketVecReal = ket.deviceStateVec.real;
  qreal *ketVecImag = ket.deviceStateVec.imag;
  Complex ans;
  ans.real =
      get_wukDeviceHandle(bra)->dot(numAmps, braVecReal, 1, ketVecReal, 1) +
      get_wukDeviceHandle(bra)->dot(numAmps, braVecImag, 1, ketVecImag, 1);
  ans.imag =
      get_wukDeviceHandle(bra)->dot(numAmps, braVecReal, 1, ketVecImag, 1) -
      get_wukDeviceHandle(bra)->dot(numAmps, braVecImag, 1, ketVecReal, 1);
  MPI_Allreduce(MPI_IN_PLACE, &ans.real, 1, MPI_QuEST_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ans.imag, 1, MPI_QuEST_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  return ans;
}

/*
 * outcomeProb must accurately be the probability of that qubit outcome in the
 * state-vector, or else the state-vector will lose normalisation
 */
void statevec_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit,
                                         int outcome, qreal outcomeProb) {
  long long int mask = 1LL << measureQubit;
  auto term1 = make_cuDoubleComplex(1 / sqrt(outcomeProb), 0),
       term2 = make_cuDoubleComplex(0, 0);
  if (outcome)
    std::swap(term1, term2);
  statevec_phaseShiftByTerm_pushTask(qureg, mask, mask, term1);
  statevec_phaseShiftByTerm_pushTask(qureg, mask, 0, term2);
}

void densmatr_initPureState(Qureg targetQureg, Qureg copyQureg) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_initPlusState(Qureg qureg) {
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.clear();
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_initClassicalState(Qureg qureg, long long int stateInd) {
  reinterpret_cast<Rhandle *>(qureg.rhandle)->tbd.clear();
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

qreal densmatr_calcTotalProb(Qureg qureg) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

qreal densmatr_findProbabilityOfZero(Qureg qureg, const int measureQubit) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
  return (qreal)0; // fake return
}

qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit,
                                 int outcome) {
  qreal outcomeProb = densmatr_findProbabilityOfZero(qureg, measureQubit);
  if (outcome == 1)
    outcomeProb = 1.0 - outcomeProb;
  return outcomeProb;
}

// @TODO implement
qreal densmatr_calcFidelity(Qureg qureg, Qureg pureState) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
  return (qreal)0; // fake return
}

/** Computes the trace of the density matrix squared */
qreal densmatr_calcPurity(Qureg qureg) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
  return (qreal)0; // fake return
}

/** This involves finding |...i...><...j...| states and killing those where i!=j
 */
void densmatr_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit,
                                         int outcome, qreal totalStateProb) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_addDensityMatrix(Qureg combineQureg, qreal otherProb,
                               Qureg otherQureg) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_oneQubitDegradeOffDiagonal(Qureg qureg, const int targetQubit,
                                         qreal dephFac) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_oneQubitDamping(Qureg qureg, const int targetQubit,
                              qreal damping) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_twoQubitDepolarise(Qureg qureg, int qubit1, int qubit2,
                                 qreal depolLevel) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_oneQubitDepolarise(Qureg qureg, const int targetQubit,
                                 qreal depolLevel) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

void densmatr_oneQubitDephase(Qureg qureg, const int targetQubit,
                              qreal dephase) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

// @TODO is separating these 12 amplitudes really faster than letting every
// 16th base modify 12 elems?
void densmatr_twoQubitDephase(Qureg qureg, int qubit1, int qubit2,
                              qreal dephase) {
  printf("densmatr series has not been supported yet.\n");
  exit(EXIT_FAILURE);
}

#if defined(__cplusplus)
}
#endif