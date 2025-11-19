# Multi-Node Setup and Configuration

## Table of Contents
1. [Multi-Node Architecture](#architecture)
2. [Network Requirements](#network)
3. [SSH and Security Setup](#ssh-setup)
4. [Environment Configuration](#environment)
5. [Distributed Process Launch](#launch)
6. [Storage and File Systems](#storage)
7. [Monitoring Multi-Node Clusters](#monitoring)
8. [Common Issues and Solutions](#troubleshooting)

---

## Multi-Node Architecture {#architecture}

### Typical Multi-Node Setup

```
┌─────────────────────────────────────────────────────┐
│                  Head/Master Node                   │
│  - Coordinates distributed training                  │
│  - Runs scheduler and load balancer                │
│  - Hosts shared filesystems (NFS/Lustre)           │
│  - 8x A100 GPUs (can also run compute)             │
└──────────────┬────────────────────────────────────┘
               │
        High-bandwidth network
      (InfiniBand or 100GbE)
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────────┐    ┌────▼──────────┐
│ Worker Node 1│    │ Worker Node 2 │
│ 8x A100 GPUs │    │ 8x A100 GPUs  │
└──────────────┘    └───────────────┘
```

### Node Roles

```python
class NodeRole:
    """Different roles in multi-node setup."""

    MASTER = "master"  # Rank 0, coordinates
    WORKER = "worker"  # Ranks 1+, compute

def get_node_role():
    """Determine this node's role."""
    import os

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if rank == 0:
        return NodeRole.MASTER
    else:
        return NodeRole.WORKER


def setup_based_on_role():
    """Setup based on node role."""
    role = get_node_role()

    if role == NodeRole.MASTER:
        # Master node responsibilities
        setup_scheduler()
        setup_logging_server()
        setup_checkpoint_manager()

    # All nodes
    setup_distributed_backend()
    load_model()
```

---

## Network Requirements {#network}

### Network Topology

```bash
# Check network topology
# Shows which network interfaces connect to which GPUs

nvidia-smi topo -m

# Output example:
#         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
# GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
# GPU1   NV12    X    NV12  NV12  NV12  NV12  NV12  NV12
# ...

# NV12 = NVLink 3.0 (300 GB/s bidirectional)
# PIX = PCIe + NVSwitch
# PHB = PCIe host bridge
```

### Network Bandwidth Requirements

```python
def calculate_bandwidth_requirements(
    model_size_gb: float,
    num_nodes: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    batch_size: int,
    seq_length: int,
):
    """
    Calculate network bandwidth requirements.
    """

    print("Network Bandwidth Requirements")
    print("=" * 60)

    # Intra-node (NVLink)
    if tensor_parallel_size > 1:
        # All-reduce communication per layer
        hidden_size = 8192  # Example
        activation_size_mb = batch_size * seq_length * hidden_size * 2 / 1e6

        # ~2 all-reduces per layer
        comm_per_layer = activation_size_mb * 2

        print("Intra-node (NVLink):")
        print(f"  Communication per layer: {comm_per_layer:.2f} MB")
        print(f"  Recommended: NVLink 3.0+ (300 GB/s)")
        print()

    # Inter-node (InfiniBand/Ethernet)
    if num_nodes > 1:
        if pipeline_parallel_size > 1:
            # Pipeline communication: activation passing
            activation_size_mb = batch_size * seq_length * hidden_size * 2 / 1e6

            print("Inter-node (Pipeline):")
            print(f"  Activation size per micro-batch: {activation_size_mb:.2f} MB")
            print(f"  Recommended: 100 GbE or HDR InfiniBand (200 GB/s)")
            print()

        # Data parallelism: gradient all-reduce
        gradient_size_gb = model_size_gb

        print("Inter-node (Data Parallelism):")
        print(f"  Gradient size: {gradient_size_gb:.2f} GB")
        print(f"  Recommended: 100 GbE minimum")


# Example: 70B model on 4 nodes
calculate_bandwidth_requirements(
    model_size_gb=140,  # FP16
    num_nodes=4,
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    batch_size=32,
    seq_length=2048,
)
```

### Network Configuration

```bash
# === InfiniBand Configuration ===

# Check IB devices
ibstat

# Test IB bandwidth between nodes
# On receiving node:
ib_write_bw

# On sending node:
ib_write_bw <receiver_hostname>

# Expected: ~190-200 GB/s for HDR IB


# === Ethernet Configuration ===

# Check network interfaces
ip addr show

# Test bandwidth with iperf3
# On receiver:
iperf3 -s

# On sender (4 parallel streams):
iperf3 -c <receiver_ip> -P 4

# Expected: ~90-95 Gbps for 100GbE


# === Configure firewall ===

# Open ports for distributed training
# PyTorch distributed: 29500-29600
# Ray: 6379, 8265, 10001-10100
# NCCL: 1024-65535 (dynamic)

sudo ufw allow 29500:29600/tcp
sudo ufw allow 6379/tcp
sudo ufw allow 8265/tcp
sudo ufw allow 10001:10100/tcp
```

---

## SSH and Security Setup {#ssh-setup}

### Passwordless SSH

```bash
# === Setup SSH keys for passwordless access ===

# On master node, generate SSH key
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Copy to all worker nodes
for node in worker1 worker2 worker3; do
    ssh-copy-id -i ~/.ssh/id_rsa.pub user@$node
done

# Test connectivity
for node in worker1 worker2 worker3; do
    ssh $node "hostname && nvidia-smi -L"
done

# === Create hostfile ===
cat > hostfile << 'EOF'
master slots=8
worker1 slots=8
worker2 slots=8
worker3 slots=8
EOF
```

### Security Best Practices

```bash
# === Secure SSH configuration ===

# Edit /etc/ssh/sshd_config
sudo tee -a /etc/ssh/sshd_config << EOF
# Disable password authentication (use keys only)
PasswordAuthentication no

# Disable root login
PermitRootLogin no

# Use strong ciphers
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com

# Key exchange algorithms
KexAlgorithms curve25519-sha256,diffie-hellman-group-exchange-sha256
EOF

sudo systemctl restart sshd


# === Firewall rules ===

# Only allow SSH from specific IPs
sudo ufw allow from 192.168.1.0/24 to any port 22

# Allow cluster communication within private network
sudo ufw allow from 192.168.1.0/24
```

---

## Environment Configuration {#environment}

### Environment Variables

```bash
# === Common environment variables for multi-node ===

# Master node address
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500

# Node information
export RANK=0  # 0 for master, 1+ for workers
export WORLD_SIZE=4  # Total number of nodes
export LOCAL_RANK=0  # GPU rank on this node
export LOCAL_WORLD_SIZE=8  # GPUs per node

# NCCL configuration
export NCCL_SOCKET_IFNAME=eth0  # Or ib0 for InfiniBand
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0  # InfiniBand device

# GPU Direct RDMA
export NCCL_NET_GDR_LEVEL=5

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### Shared Configuration File

```python
# config.yaml - Shared across all nodes

cluster:
  nodes:
    - hostname: master
      ip: 192.168.1.100
      gpus: 8
    - hostname: worker1
      ip: 192.168.1.101
      gpus: 8
    - hostname: worker2
      ip: 192.168.1.102
      gpus: 8

  network:
    master_addr: 192.168.1.100
    master_port: 29500
    backend: nccl
    interface: ib0  # InfiniBand interface

model:
  name: meta-llama/Llama-2-70b-hf
  tensor_parallel_size: 8
  pipeline_parallel_size: 1

paths:
  model_cache: /shared/models
  checkpoint_dir: /shared/checkpoints
  log_dir: /shared/logs


# Load configuration
import yaml

def load_cluster_config(config_path="/shared/config.yaml"):
    """Load cluster configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set environment variables
    os.environ['MASTER_ADDR'] = config['cluster']['network']['master_addr']
    os.environ['MASTER_PORT'] = str(config['cluster']['network']['master_port'])

    return config
```

---

## Distributed Process Launch {#launch}

### Using torchrun

```bash
# === Single-node multi-GPU ===
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py


# === Multi-node setup ===

# On master node (rank 0):
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py

# On worker node 1 (rank 1):
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py

# On worker node 2 (rank 2):
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=2 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py
```

### Using MPI (OpenMPI)

```bash
# === MPI-based launch ===

# Install OpenMPI with CUDA support
sudo apt-get install openmpi-bin libopenmpi-dev

# Create hostfile
cat > hosts << EOF
master slots=8
worker1 slots=8
worker2 slots=8
EOF

# Launch with mpirun
mpirun -np 24 \
    -H master:8,worker1:8,worker2:8 \
    --bind-to none \
    --map-by slot \
    -x MASTER_ADDR=master \
    -x MASTER_PORT=29500 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE=0 \
    python train.py
```

### Using Ray

```bash
# === Ray cluster launch ===

# On head node:
ray start \
    --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --num-gpus=8

# On each worker node:
ray start \
    --address='192.168.1.100:6379' \
    --num-gpus=8

# Verify cluster
ray status


# Launch vLLM on Ray
python -c "
from vllm import LLM

llm = LLM(
    model='meta-llama/Llama-2-70b-hf',
    tensor_parallel_size=8,
    # Ray automatically uses all nodes
)

outputs = llm.generate(['Hello world'])
print(outputs[0].outputs[0].text)
"
```

### Launch Script

```python
#!/usr/bin/env python3
"""
launch_multinode.py - Launch distributed training on multiple nodes
"""

import subprocess
import argparse
from typing import List

def launch_on_nodes(
    nodes: List[str],
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    script: str,
):
    """
    Launch training on multiple nodes using SSH.
    """

    processes = []

    for node_rank, node in enumerate(nodes):
        # Build command
        if node_rank == 0:
            # Master node - run locally
            cmd = [
                "torchrun",
                f"--nproc_per_node={gpus_per_node}",
                f"--nnodes={len(nodes)}",
                f"--node_rank={node_rank}",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                script,
            ]

            proc = subprocess.Popen(cmd)
        else:
            # Worker nodes - run via SSH
            cmd = [
                "ssh",
                node,
                f"torchrun "
                f"--nproc_per_node={gpus_per_node} "
                f"--nnodes={len(nodes)} "
                f"--node_rank={node_rank} "
                f"--master_addr={master_addr} "
                f"--master_port={master_port} "
                f"{script}"
            ]

            proc = subprocess.Popen(cmd)

        processes.append(proc)

    # Wait for all to complete
    for proc in processes:
        proc.wait()

    print("All nodes completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--script", required=True)

    args = parser.parse_args()

    launch_on_nodes(
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port,
        script=args.script,
    )


# Usage:
# python launch_multinode.py \
#     --nodes master worker1 worker2 \
#     --gpus-per-node 8 \
#     --master-addr 192.168.1.100 \
#     --script train.py
```

---

## Storage and File Systems {#storage}

### Shared File System Setup

```bash
# === NFS Setup (Simple, lower performance) ===

# On master node (NFS server):
sudo apt-get install nfs-kernel-server

# Create shared directory
sudo mkdir -p /shared
sudo chown nobody:nogroup /shared

# Export directory
echo "/shared *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports

# Restart NFS
sudo systemctl restart nfs-kernel-server


# On worker nodes (NFS clients):
sudo apt-get install nfs-common

# Create mount point
sudo mkdir -p /shared

# Mount NFS share
sudo mount master:/shared /shared

# Make permanent (add to /etc/fstab)
echo "master:/shared /shared nfs defaults 0 0" | sudo tee -a /etc/fstab


# === Lustre (High-performance, for HPC) ===
# Lustre provides parallel filesystem for large-scale clusters
# Setup is complex, typically handled by cluster admins
```

### Model and Checkpoint Storage

```python
import os
from pathlib import Path

class DistributedStorage:
    """Manage storage in distributed setup."""

    def __init__(self, shared_dir="/shared"):
        self.shared_dir = Path(shared_dir)
        self.model_cache = self.shared_dir / "models"
        self.checkpoint_dir = self.shared_dir / "checkpoints"
        self.log_dir = self.shared_dir / "logs"

        # Create directories (only on rank 0)
        if self.is_master():
            self.model_cache.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Synchronize
        self.barrier()

    def is_master(self):
        """Check if this is the master node."""
        return int(os.environ.get('RANK', 0)) == 0

    def barrier(self):
        """Synchronization barrier."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def get_model_path(self, model_name):
        """Get path for cached model."""
        return self.model_cache / model_name

    def save_checkpoint(self, state, epoch):
        """Save checkpoint (only on rank 0)."""
        if self.is_master():
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(state, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        self.barrier()

    def load_checkpoint(self, epoch):
        """Load checkpoint (all ranks)."""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"

        # Ensure file exists before loading
        self.barrier()

        if checkpoint_path.exists():
            return torch.load(checkpoint_path)
        else:
            return None


# Usage
storage = DistributedStorage("/shared")

# Save model (only rank 0 writes)
storage.save_checkpoint({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, epoch)

# Load model (all ranks read)
checkpoint = storage.load_checkpoint(epoch)
```

---

## Monitoring Multi-Node Clusters {#monitoring}

### Cluster Monitoring

```python
import psutil
import torch

def monitor_cluster():
    """Monitor multi-node cluster resources."""

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Gather local statistics
    stats = {
        'rank': rank,
        'hostname': socket.gethostname(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'gpu_memory_allocated': torch.cuda.max_memory_allocated() / 1e9,
        'gpu_utilization': get_gpu_utilization(),
    }

    # Gather from all ranks
    all_stats = [None] * world_size
    dist.all_gather_object(all_stats, stats)

    # Print on rank 0
    if rank == 0:
        print("\nCluster Status:")
        print("=" * 80)

        for node_stats in all_stats:
            print(f"\nNode {node_stats['rank']} ({node_stats['hostname']}):")
            print(f"  CPU: {node_stats['cpu_percent']:.1f}%")
            print(f"  Memory: {node_stats['memory_percent']:.1f}%")
            print(f"  GPU Memory: {node_stats['gpu_memory_allocated']:.2f} GB")
            print(f"  GPU Util: {node_stats['gpu_utilization']:.1f}%")


def get_gpu_utilization():
    """Get GPU utilization using nvidia-ml-py."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pynvml.nvmlShutdown()

    return utilization.gpu
```

---

## Common Issues and Solutions {#troubleshooting}

### Issue 1: Network Timeout

```bash
# Symptom: NCCL timeout errors across nodes

# Solution 1: Increase timeout
export NCCL_TIMEOUT=1800  # 30 minutes

# Solution 2: Check network connectivity
ping worker1
ping worker2

# Solution 3: Test NCCL bandwidth
# On each node:
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi

# Run test across nodes
mpirun -np 16 -H master:8,worker1:8 ./build/all_reduce_perf -b 8 -e 1G -f 2
```

### Issue 2: SSH Hangs

```bash
# Symptom: SSH connections hang or time out

# Solution: Increase SSH timeout and keep-alive
echo "ServerAliveInterval 60" >> ~/.ssh/config
echo "ServerAliveCountMax 3" >> ~/.ssh/config
```

### Issue 3: File System Synchronization

```python
def ensure_file_sync():
    """Ensure file system is synchronized across nodes."""

    rank = dist.get_rank()

    # Master writes file
    if rank == 0:
        with open("/shared/ready.txt", "w") as f:
            f.write("ready")

        # Force sync
        os.sync()

    # All ranks wait
    dist.barrier()

    # Workers verify file exists
    max_retries = 10
    for i in range(max_retries):
        if os.path.exists("/shared/ready.txt"):
            break
        time.sleep(1)
    else:
        raise RuntimeError("File not synchronized")

    # Clean up
    if rank == 0:
        os.remove("/shared/ready.txt")
```

---

## Summary

Multi-node setup enables large-scale distributed inference:

**Key Requirements:**
- High-bandwidth network (InfiniBand or 100GbE)
- Shared file system (NFS, Lustre)
- Passwordless SSH
- Proper environment configuration

**Launch Methods:**
- torchrun (PyTorch native)
- MPI (traditional HPC)
- Ray (modern distributed framework)

**Best Practices:**
- Test network thoroughly before deployment
- Use shared storage for models and checkpoints
- Monitor cluster health regularly
- Handle failures gracefully

Continue to the next module on distributed KV cache management.
