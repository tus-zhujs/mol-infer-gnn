# Trained models

The 16 raw PyTorch checkpoints used in Phase 2 are included in `checkpoints/`.
They total **146,702,942 bytes** (about 146.70 MB); each file is below GitHub's
100 MiB regular-file limit. Checkpoint bytes were checked against the original
`data/phase1/asset-inventory.json` before and after copying. No weights were
converted or loaded during packaging.

The original files remain under `data/phase1/objects/checkpoints/sha256/` in the
experiment repository, at `<first-two-hash-characters>/<sha256>.pt` beneath
that directory. The table gives the exact SHA-256 identifiers. The other
20 Phase 1 checkpoints are not included.

The corresponding 36 small original configurations are in `configs/`, named
by model ID. These include the 16 configurations for the checkpoints below.
The historical loader also expects the original normalization/inventory
metadata and experiment layout; this is not a standalone inference bundle.

| Checkpoint | Property | Architecture | Bytes | SHA-256 |
|---|---|---|---:|---|
| [journal_homo_2lay_h16](checkpoints/journal_homo_2lay_h16.pt) | homo | MP2-16 | 9,160,628 | `17b0b584b4addf917970cbcdda4e54ffafefb3173e8265e3ef5c49a6d149fac0` |
| [journal_homo_2lay_h32](checkpoints/journal_homo_2lay_h32.pt) | homo | MP2-32 | 9,170,868 | `df8ba28724a8aceb5683bce50a406fbe8e7ff600304c7355d5a83b51b85e60fe` |
| [journal_homo_3lay_h16](checkpoints/journal_homo_3lay_h16.pt) | homo | MP3-16 | 9,163,937 | `b19e4e75a0c6d37e05600822d76954b0c2a70d8210ef8cc3b6704c6ccd70f982` |
| [journal_homo_3lay_h32](checkpoints/journal_homo_3lay_h32.pt) | homo | MP3-32 | 9,180,385 | `ed004babec88d1350e34cd87336cc178653f7ce71b1e23aa70f888393939bcf0` |
| [journal_lumo_2lay_h16](checkpoints/journal_lumo_2lay_h16.pt) | lumo | MP2-16 | 9,160,628 | `ff96f8d0bb2ba6f86081c7193b797e28ec49e887331ee91f7326b6ada818dc60` |
| [journal_lumo_2lay_h32](checkpoints/journal_lumo_2lay_h32.pt) | lumo | MP2-32 | 9,170,868 | `73241e0fa46d3fb3503c5d095071d8f1ce934583b7b7ccdb5b926fbaa4bfc66a` |
| [journal_lumo_3lay_h16](checkpoints/journal_lumo_3lay_h16.pt) | lumo | MP3-16 | 9,163,937 | `a3c7ce3f76086b39c513be4036d17112e854b8baccaab38222a2dd5c2bee7538` |
| [journal_lumo_3lay_h32](checkpoints/journal_lumo_3lay_h32.pt) | lumo | MP3-32 | 9,180,385 | `b922273651477b7cc011170438d7c99e767803d4b487a11dc16d604e2c10283d` |
| [journal_gap_2lay_h16](checkpoints/journal_gap_2lay_h16.pt) | gap | MP2-16 | 9,160,547 | `7b7866a60037936547c5a535ba853e33a66f4b136849a80b22550f45af2d81c1` |
| [journal_gap_2lay_h32](checkpoints/journal_gap_2lay_h32.pt) | gap | MP2-32 | 9,170,787 | `5b5c4590bdaa841b63ff3d88b4a398d347688161edd0e320a08a59e2f2bbdff7` |
| [journal_gap_3lay_h16](checkpoints/journal_gap_3lay_h16.pt) | gap | MP3-16 | 9,163,853 | `e464035878f60e63adf0e026e740cd77fda434f46ec482dfee43b01170f60566` |
| [journal_gap_3lay_h32](checkpoints/journal_gap_3lay_h32.pt) | gap | MP3-32 | 9,180,301 | `844dd99455817be6ed82b47519627182671657d77e6640e943fb7fe6a0f7ad19` |
| [journal_zpve_2lay_h16](checkpoints/journal_zpve_2lay_h16.pt) | zpve | MP2-16 | 9,160,628 | `e9180e000d333536642dd2de038d98a3c76d461a17ec78e47e9c64218b1b2414` |
| [journal_zpve_2lay_h32](checkpoints/journal_zpve_2lay_h32.pt) | zpve | MP2-32 | 9,170,868 | `904a3387715c00b130d237fa41c3f6019a77003176ebe930b8215f64013287fa` |
| [journal_zpve_3lay_h16](checkpoints/journal_zpve_3lay_h16.pt) | zpve | MP3-16 | 9,163,937 | `5304133a81715fb8f00536bd247aa18acad8bbd4b06747f8aa467fdaa4dae5ee` |
| [journal_zpve_3lay_h32](checkpoints/journal_zpve_3lay_h32.pt) | zpve | MP3-32 | 9,180,385 | `2bb026bb7d9d0e699378caf2342efad76dc946eab53252bbc4c1f24b83982bab` |
