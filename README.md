# BlueDisc: Adversarial Shape Learning for Seismic Phase Picking

This repo is a minimal, reproducible implementation to validate the paper “Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning.” It augments a PhaseNet generator with a lightweight conditional discriminator (BlueDisc) to enforce label shape learning, which eliminates the 0.5-amplitude suppression band and increases effective S-phase detections.

## Pre-release

This is a pre-release version. The code will be cleaned up and documented further in the near future.

## Citation

Please cite the paper when using this code:

```bibtex
@article{huang2025bluedisc,
  title={Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning},
  author={Chun-Ming Huang and Li-Heng Chang and I-Hsin Chang and An-Sheng Lee and Hao Kuo-Chen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## References

Key papers referenced in this work:

- **PhaseNet**: Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261-273.  
  DOI: [10.1093/gji/ggy423](https://doi.org/10.1093/gji/ggy423)

- **GAN**: Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative adversarial nets. *NeurIPS*.  
  [Paper](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html) | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

- **Conditional GAN**: Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*.  
  [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)

- **pix2pix**: Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *CVPR*.  
  DOI: [10.1109/CVPR.2017.632](https://doi.org/10.1109/CVPR.2017.632) | [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)

- **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *MICCAI*.  
  DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28) | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

- **SeisBench**: Woollam, J., Rietbrock, A., Bueno, A., & De Angelis, S. (2022). SeisBench—A toolbox for machine learning in seismology. *Seismological Research Letters*, 93(3), 1695-1709.  
  DOI: [10.1785/0220210324](https://doi.org/10.1785/0220210324) | [GitHub](https://github.com/seisbench/seisbench)

- **Pick-Benchmark**: Münchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2022). Which picker fits my data? A quantitative evaluation of deep learning based seismic pickers. *JGR: Solid Earth*, 127(1).  
  DOI: [10.1029/2021JB023499](https://doi.org/10.1029/2021JB023499) | [GitHub](https://github.com/seisbench/pick-benchmark)

- **INSTANCE Dataset**: Michelini, A., Cianetti, S., Gaviano, S., et al. (2021). INSTANCE–the Italian seismic dataset for machine learning. *Earth System Science Data*, 13(12), 5509-5544.  
  DOI: [10.5194/essd-13-5509-2021](https://doi.org/10.5194/essd-13-5509-2021)

