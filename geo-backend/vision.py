# vision.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

CATEGORIES = {
    "vegetation": [
        "Mediterranean shrubland with olive and cypress trees",
        "dense temperate deciduous forest",
        "coniferous alpine forest",
        "dry Australian eucalyptus bushland",
        "tropical palm and banana vegetation",
        "African savanna grassland with acacia trees",
        "Brazilian cerrado with scattered trees and red soil",
        "flat farmland with hedgerows",
        "sparse scrubby vegetation in arid climate",
        "dense hillside jungle or monsoon forest, broadleaf canopy without visible palms",
    ],
    "architecture": [
        "German or Austrian half-timbered houses",
        "French rural stone farmhouses",
        "Alpine wooden chalets with sloped roofs",
        "British brick terraced houses",
        "American suburban houses with front lawns and driveways",
        "colorful Latin American concrete buildings",
        "Eastern European Soviet-era apartment blocks",
        "no visible buildings, open countryside",
    ],
    "terrain": [
        "flat open plains stretching to the horizon",
        "rolling green hills",
        "steep snow-capped alpine mountains",
        "rocky coastal cliffs",
        "dense forest with no visible horizon",
        "open rural landscape with scattered trees and wide sky",
        "arid desert with sparse vegetation",
        "river valley surrounded by hills",
    ],
    "road_infrastructure": [
        "narrow rural two-lane road with no markings",
        "wide divided highway with metal guardrails",
        "unpaved reddish dirt road through rural countryside",
        "road with a solid yellow center line",
        "road with a white dashed center line",
        "cobblestone or brick paved street",
        "gravel or unpaved dirt road",
        "narrow paved two-lane road cutting through dense forest or jungle",
    ],
    "signage_style": [
        "road signs with a blue background typical of France",
        "road signs with a yellow background typical of Germany",
        "road signs with a white background and black border",
        "road signs written in a non-Latin script",
        "no visible road signs",
    ],
    "boundary_style": [
        "high concrete or brick wall topped with razor wire or electric fencing",
        "low decorative garden wall or hedge",
        "plain wire or post-and-rail rural fencing",
        "no visible boundary wall or fence",
        "chain-link fencing",
    ],
}

_category_names = list(CATEGORIES.keys())
_flat_labels = []
_category_slices = {}

for cat in _category_names:
    start = len(_flat_labels)
    _flat_labels.extend(CATEGORIES[cat])
    _category_slices[cat] = (start, len(_flat_labels))

with torch.no_grad():
    _label_inputs = processor(text=_flat_labels, return_tensors="pt", padding=True)
    _label_embeds = model.get_text_features(**_label_inputs)
    if hasattr(_label_embeds, "pooler_output"):
        _label_embeds = _label_embeds.pooler_output
    elif hasattr(_label_embeds, "last_hidden_state"):
        _label_embeds = _label_embeds.last_hidden_state[:, 0, :]
    _label_embeds = _label_embeds / _label_embeds.norm(dim=-1, keepdim=True)


def analyze_image(image: Image.Image, top_k=3):
    # Encode the image exactly once.
    img_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_embed = model.get_image_features(**img_inputs)
        if hasattr(img_embed, "pooler_output"):
            img_embed = img_embed.pooler_output
        elif hasattr(img_embed, "last_hidden_state"):
            img_embed = img_embed.last_hidden_state[:, 0, :]
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        sims = (img_embed @ _label_embeds.T).squeeze(0)

    results = {}
    for cat in _category_names:
        start, end = _category_slices[cat]
        cat_sims = sims[start:end]
        cat_probs = torch.softmax(cat_sims * model.logit_scale.exp(), dim=0)

        ranked = sorted(
            zip(CATEGORIES[cat], cat_probs.tolist()),
            key=lambda x: -x[1]
        )
        results[cat] = [
            {"label": label, "confidence": round(score, 3)}
            for label, score in ranked[:top_k]
        ]

    return results

if __name__ == "__main__":
    import sys
    import json

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    image = Image.open(image_path).convert("RGB")
    results = analyze_image(image)
    print(json.dumps(results, indent=2))