## Can Score-based Generative Modeling Effectively Handle Medical Image Classification?

Discriminative models often face a trade-off between data likelihood and classification accuracy. In this study, we investigate the use of score-based generative models as classifiers for medical images, with a focus on mammographic images. By leveraging their ability to capture underlying data distributions in closely related classes and achieve competitive likelihood values, we aim to address the limitations of discriminative models and highlight their potential for improved performance.

### Figures

<img src="https://github.com/user-attachments/assets/60b76aa5-5ee0-47b2-91c0-355420299ac2" alt="image" width="650" height="400" />




### Results

Table 1: Evaluation of various architectural configurations in the proposed approach across three datasets  
*Acc.: Accuracy, AUC: Area under the curve*

| SDE Function | CBIS Acc. | CBIS AUC | INbreast Acc. | INbreast AUC | Vin-Dr Acc. | Vin-Dr AUC |
|--------------|-----------|----------|----------------|---------------|--------------|-------------|
| **VPSDE**    | **63.65** | **71.75** | **75.00**     | **78.85**     | 84.78        | **64.16**   |
| VESDE        | 62.60     | 54.01     | 63.64         | 42.85         | 84.92        | 48.77       |
| SubVPSDE     | 58.73     | 51.69     | 63.64         | 36.5          | **85.72**    | 50.78       |

Table 2: Comparative assessment of the proposed architecture with state-of-the-art discriminative models  
*Acc.: Accuracy, Spe.: Specificity, Sen.: Sensitivity*

| Dataset        | Metrics | ResNet50 | InceptionV3 | VIT    | Swin-T | Ours          |
|----------------|---------|----------|-------------|--------|--------|---------------|
| **CBIS**       | Acc.    | 54.35    | 59.54       | 58.20  | 55.96  | **63.65**     |
|                | Spe.    | **80.00**| 58.10       | 58.40  | 51.39  | 72.83         |
|                | Sen.    | 47.22    | 41.50       | 44.10  | 41.38  | **58.21**     |
|                | AUC     | 61.38    | 47.78       | 52.44  | 52.41  | **71.75**     |
| **INbreast**   | Acc.    | 45.65    | 66.52       | 56.41  | 68.56  | **75.00**     |
|                | Spe.    | 37.60    | 30.27       | 37.10  | 20.00  | **44.44**     |
|                | Sen.    | 65.30    | 70.50       | 66.70  | 78.60  | **82.86**     |
|                | AUC     | 50.83    | 60.78       | 51.87  | 66.66  | **78.85**     |
| **VinDr-Mammo**| Acc.    | 52.20    | 75.96       | 64.01  | 84.40  | **87.78**     |
|                | Spe.    | 51.95    | **92.24**   | 49.81  | 87.87  | 84.78         |
|                | Sen.    | **57.07**| 37.77       | 52.34  | 17.68  | 44.14         |
|                | AUC     | 57.59    | 58.29       | 51.01  | 53.67  | **64.16**     |


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{sarker2025can,
  title={Can Score-Based Generative Modeling Effectively Handle Medical Image Classification?},
  author={Sarker, Sushmita and Sarker, Prithul and Bebis, George and Tavakkoli, Alireza},
  journal={arXiv preprint arXiv:2502.17727},
  year={2025}
}


