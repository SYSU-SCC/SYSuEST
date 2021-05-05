# SYSuEST

Distributed GPU acceleration of [QuEST_v2.1.0](https://github.com/QuEST-Kit/QuEST/releases/tag/2.1.0) for the QuEST Chanllenge in [ASC20-21](http://www.asc-events.net/ASC20-21/Preliminary.php).

Supported by Student Cluster Competition Team @ Sun Yat-sen University.

## Quick Start

```bash
bash examples/SYSuEST/run.sh
```

## Results

|  GPU(s)   |   1   |   2   |   4   |
| :-------: | :---: | :---: | :---: |
|  random   |   x   | 9.82s | 7.48s |
|    GHZ    |   x   | 1.13s | 0.86s |
| GHZ_QFT_N | 0.91s | 0.54s | 0.43s |

### Environment

```bash
$ nvidia-smi
Wed May  5 18:58:36 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:8A:00.0 Off |                    0 |
| N/A   39C    P0    56W / 300W |     10MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:8B:00.0 Off |                    0 |
| N/A   36C    P0    61W / 300W |     10MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:B3:00.0 Off |                    0 |
| N/A   36C    P0    62W / 300W |     10MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:B4:00.0 Off |                    0 |
| N/A   39C    P0    60W / 300W |     10MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    mlx5_0  mlx5_1  mlx5_2  mlx5_3  CPU Affinity
GPU0     X      NV2     NV1     NODE    PIX     PIX     NODE    NODE    14-27
GPU1    NV2      X      NODE    NV2     PIX     PIX     NODE    NODE    14-27
GPU2    NV1     NODE     X      NV2     NODE    NODE    PIX     PIX     14-27
GPU3    NODE    NV2     NV2      X      NODE    NODE    PIX     PIX     14-27
mlx5_0  PIX     PIX     NODE    NODE     X      PIX     NODE    NODE
mlx5_1  PIX     PIX     NODE    NODE    PIX      X      NODE    NODE
mlx5_2  NODE    NODE    PIX     PIX     NODE    NODE     X      PIX
mlx5_3  NODE    NODE    PIX     PIX     NODE    NODE    PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
```
