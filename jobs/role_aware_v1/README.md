# HALSP role-aware v1 frozen job

This data-only job binds the experiment to HALSP commit
`087725a74be5407d750c537ac701d82531c68a91` and the pinned Hugging Face
CIFAR-100 revision. It exposes no commands, scripts, modules, or entry points.

The fixed handler runs six Stage-A endpoints: full-support probe momentum,
structural Taylor, and role-aware Taylor on seeds 0 and 1. Layer1/layer2 retain
the pinned dense architecture; only layer3/layer4 use exact 50% Focus.

An initial deterministic, augment-free, 1,000-example training-only probe at
epoch 6 selects the first exact-K masks. Five material probes occur at epochs
20, 40, 60, 80, and 100. Stage B opens only when both RAT seeds have at least
three material sparse-stage checkpoints and epoch 100 is material. It then
adds RAT plus two-DOF role-specific modulation and the equal-capacity role-blind
control on both seeds. The frozen maximum is ten endpoints.

No Explore, Reserve exposure update, hard swap, or test-set decision is allowed.
Official CIFAR-100 test evaluation begins only after the Stage-B decision is
written and uses the frozen zero-based epoch-100 sparse checkpoint.
