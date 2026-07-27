# GeoCoach

A location classifier by country that predicts where a street-view photo was taken,
built on DINOv2 embeddings and CLIP-based visual reasoning.

## Overview

GeoCoach predicts a photo's country of origin by combining:
- **DINOv2** embeddings (1536-dim) fed into a logistic regression classifier, trained 
  on 234,000 street-view images across 117 countries
- **CLIP-based** visual reasoning for natural context clues
- A candidate-narrowing pipeline that combines both signals with confidence weighting

## Performance

**64.7% top-1 accuracy** across 117 countries, evaluated on a held-out test set of 
46,552 images (split by location sequence to prevent leakage between near-duplicate shots).

## Stack

- **Frontend**: Next.js
- - **ML**: Python, DINOv2, CLIP, scikit-learn
- **Retraining**: `retrain.py` supports versioning from user feedback

## Acknowledgments

Trained using embeddings from [DINOv2](https://github.com/facebookresearch/dinov2) and 
[CLIP](https://github.com/openai/CLIP).

Training data from [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) 
(Astruc et al., CVPR 2024), licensed under CC-BY-SA-4.0.

\`\`\`bibtex
@article{osv5m,
    title = {{OpenStreetView-5M}: {T}he Many Roads to Global Visual Geolocation},
    author = {Astruc, Guillaume and Dufour, Nicolas and Siglidis, Ioannis
      and Aronssohn, Constantin and Bouia, Nacim and Fu, Stephanie and Loiseau, Romain
      and Nguyen, Van Nguyen and Raude, Charles and Vincent, Elliot and Xu, Lintao
      and Zhou, Hongyu and Landrieu, Loic},
    journal = {CVPR},
    year = {2024},
}
\`\`\`
