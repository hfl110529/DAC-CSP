# Dependencies and method attribution

The release reorganizes the project's CSP implementation and shared-Q/V baseline adaptations into portable modules. Public method names and the following upstream references are retained for attribution.

| Component | Reference |
| --- | --- |
| PyTorch | https://pytorch.org/ |
| Hugging Face Transformers | https://github.com/huggingface/transformers |
| Qwen3 model family | https://huggingface.co/Qwen/Qwen3-1.7B |
| Qwen2 model family | https://huggingface.co/Qwen/Qwen2-0.5B |
| GPM | *Gradient Projection Memory for Continual Learning*, ICLR 2021 |
| O-LoRA | *Orthogonal Subspace Learning for Language Model Continual Learning*, Findings of EMNLP 2023 |
| InfLoRA | *InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning*, CVPR 2024 |
| EWC | *Overcoming Catastrophic Forgetting in Neural Networks*, PNAS 2017 |

Baseline adaptations use the paper's common trace interface, shared classifier, and Q/V update locations. GPM updates full Q/V weights, O-LoRA regularizes overlapping factors, and InfLoRA fixes the designed input factor. The code and configurations expose these differences.

Pretrained model weights and third-party source archives are not redistributed here. Install dependencies from their upstream distributions and observe their licenses and the model publishers' terms. The repository does not replace the licenses of those external components.
