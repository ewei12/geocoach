## GeoCoach

Predicts where a photo was taken using natural surroundings, combining DINOv3 visual embeddings with CLIP-based contextual reasoning, and explains the reasoning behind the guess.

## Overview

Predicts a photo's country of origin by combining:

- **DINOv3** embeddings (1536-dim) fed into an MLP classifier, trained
  on 449,154 street-view images across 117 countries
- **CLIP-based** visual reasoning for natural context clues
- A candidate-narrowing pipeline that combines both signals with confidence weighting

Inference runs on Modal's serverless platform, so embeddings are computed live for every upload.

## Performance

**76.2% top-1 accuracy** (the model's single best guess was correct) and 0.761 macro-F1 across 117 countries, evaluated on a held-out test set of
45,933 images (split by location sequence to prevent leakage between near-duplicate shots).

**Data scaling**: Doubling the training data from 234K to 449K images raised accuracy from 72.4% to 76.2% on the same held-out test set.

**Model comparison on the earlier 234k image dataset**: Tested logistic regression, an MLP, and XGBoost on the same DINOv3 embeddings and train/test split. The MLP performed best (73.0% accuracy and 0.730 macro-F1), followed by logistic regression (70.1% accuracy and 0.702 macro-F1), with XGBoost performing the worst (62.2% accuracy and 0.621 macro-F1).

Classification errors were mostly for visually similar country clusters rather than spreading evenly across all 117 countries.

## Architecture

- **Frontend**: Next.js, deployed on Vercel
- **Backend**: Python/Flask, deployed on Modal
- **Database**: Postgres on Neon - stores feedback/corrections, feeding into a retraining file (`retrain.py`)
- **ML**: DINOv3 embeddings + CLIP visual reasoning + MLP classifier, scikit-learn

## Acknowledgments

Trained using embeddings from [DINOv3](https://github.com/facebookresearch/dinov3) and
[CLIP](https://github.com/openai/CLIP).

Training data from [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m)
(Astruc et al., CVPR 2024), licensed under CC-BY-SA-4.0.

```bibtex
@article{osv5m,
    title = {{OpenStreetView-5M}: {T}he Many Roads to Global Visual Geolocation},
    author = {Astruc, Guillaume and Dufour, Nicolas and Siglidis, Ioannis
      and Aronssohn, Constantin and Bouia, Nacim and Fu, Stephanie and Loiseau, Romain
      and Nguyen, Van Nguyen and Raude, Charles and Vincent, Elliot and Xu, Lintao
      and Zhou, Hongyu and Landrieu, Loic},
    journal = {CVPR},
    year = {2024},
}
```