# Detected Codespaces Development Environment

Recorded on 2026-07-21 before the Smoke implementation and training run.

| Resource | Detected value |
|---|---:|
| Logical CPUs (`nproc`) | 9 |
| CPU | AMD EPYC 9V74 (virtualized x86_64) |
| RAM | 15 GiB total, approximately 14 GiB available at audit time |
| Swap | 0 B |
| Workspace filesystem | 63 GiB total, 53 GiB free after dependency installation |
| Python | 3.12.13 |
| NVIDIA tooling | `nvidia-smi` not installed |
| PyTorch used for verification | 2.13.0+cpu |
| CUDA available | No |
| Selected profile | Smoke / CPU |

The Codespaces development limits in `configs/smoke.yaml` are stricter than the detected host: two data workers, 4 GiB planned RAM, 8 GiB planned artifact storage, a 10-minute training deadline, and three retained candidate checkpoints. The observed 100-step experiment took 5.58 seconds, excluding environment setup.

This is a cloud development environment, not Cyrus's final local runtime. The process-level offline policy was tested here. Genuine air-gap verification must be repeated later on the owner's physical machine with its network interface or network namespace disabled.

Run `cyrus doctor --config configs/smoke.yaml` after every machine change. GPU, ROCm, CUDA, MPS, thermals, sustained disk throughput, and power-loss recovery remain unverified on physical hardware.

