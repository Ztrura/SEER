<div align="center">

## SEER: Skill-Evolving Grounded Reasoning for Free-Text Promptable 3D Medical Image Segmentation

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2603.08215)
[![Demo](https://img.shields.io/badge/Project-Demo%20Page-red?logo=youtube&logoColor=white)](https://seer-medseg.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

### 📖 Abstract

Free-text promptable 3D medical image segmentation offers an intuitive and clinically flexible interaction paradigm. However, current methods are highly sensitive to linguistic variability: minor changes in phrasing can cause substantial performance degradation despite identical clinical intent. Existing approaches attempt to improve robustness through stronger vision-language fusion or larger vocabularies, yet they lack mechanisms to consistently align ambiguous free-form expressions with anatomically grounded representations.

We propose **S**kill-**E**volving ground**E**d **R**easoning (**SEER**), a novel framework for free-text promptable 3D medical image segmentation that explicitly bridges linguistic variability and anatomical precision through a reasoning-driven design. First, we curate the SEER-Trace dataset, which pairs raw clinical requests with image-grounded, skill-tagged reasoning traces, establishing a reproducible benchmark. Second, SEER constructs an evidence-aligned target representation via a vision-language reasoning chain that verifies clinical intent against image-derived anatomical evidence, thereby enforcing semantic consistency before voxel-level decoding. Third, we introduce SEER-Loop, a dynamic skill-evolving strategy that distills high-reward reasoning trajectories into reusable skill artifacts and progressively integrates them into subsequent inference, enabling structured self-refinement and improved robustness to diverse linguistic expressions.

Extensive experiments demonstrate superior performance of SEER over state-of-the-art baselines. Under linguistic perturbations, SEER reduces performance variance by 81.94% and improves worst-case Dice by 18.60%.

### 🚀 News & TODO

We are currently organizing the codebase to ensure a clean and reproducible open-source release. Stay tuned!

- [ ] **[Coming Soon]** Release inference code and pre-trained models.
- [ ] **[Coming Soon]** Release the training scripts and dataset preparation guidelines.
- [x] **[2026-05]** Paper is early accepted by MICCAI 2026.


### 🔗 Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@article{zhang2026seer,
      title     = {Skill-Evolving Grounded Reasoning for Free-Text Promptable 3D Medical Image Segmentation},
      author    = {Zhang, Tongrui and Wang, Chenhui and Li, Yongming and Chen, Zhihao and Zhan, Xufeng and Shan, Hongming},
      journal   = {arXiv preprint arXiv:2603.08215},
      year      = {2026}
    }