# R4RR Architecture Transfer

This folder contains R4RR student-backbone transfer experiments. The goal is to keep the original R4RR teacher-map setup and evaluation protocol, while swapping the student architecture.

Current contents:

```text
architecture_transfer/
├── mobilenetv2/
└── vit/
```

The current transfer includes ViT and MobileNetV2 support for Waterbirds95,
Waterbirds100, and RedMeat. DecoyMNIST is intentionally not included in these
architecture-transfer passes.

See [vit/README.md](vit/README.md) for runnable ViT commands and file descriptions.
See [mobilenetv2/README.md](mobilenetv2/README.md) for runnable MobileNetV2
commands and file descriptions.
