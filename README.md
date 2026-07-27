## Overview

GeoCoach predicts a photo's country of origin by combining:
- **DINOv2** embeddings (1536-dim) fed into a logistic regression classifier, trained 
  on 234,000 street-view images across 117 countries
- **CLIP-based** visual reasoning for natural context clues
- A candidate-narrowing pipeline that combines both signals with confidence weighting

Inference runs on serverless GPU via Modal, so embeddings are computed live per upload.

## Performance

**64.7% top-1 accuracy** (the model's single best guess was correct) across 117 countries, evaluated on a held-out test set of 
46,552 images (split by location sequence to prevent leakage between near-duplicate shots).

## Architecture

- **Frontend**: Next.js, deployed on Vercel
- **Backend**: Python, deployed on Modal (serverless GPU inference for DINOv2/CLIP)
- **Database**: Postgres on Neon — stores feedback/corrections from the confirm/correct 
  flow, feeding into retraining (`retrain.py`)
- **ML**: DINOv2 embeddings + CLIP visual reasoning + logistic regression, scikit-learn

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
