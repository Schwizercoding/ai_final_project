# ai_final_project
# Food101 Image Classification (Transfer Learning & Zero-Shot)

## Project Overview
This project focuses on classifying food images from the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using two approaches:
1. A **custom fine-tuned ViT model**
2. A **zero-shot CLIP model** for comparison

## URLs
| Resource        | Link |
|----------------|------|
| HuggingFace Space | [Food101 Classifier Space](https://huggingface.co/spaces/jarinschnierl/food101-vit) |
| HuggingFace Model | [ViT Food101 Model](https://huggingface.co/jarinschnierl/vit-base-food101) |
| Zero-Shot Notebook | [Zero-Shot Notebook](https://huggingface.co/spaces/jarinschnierl/clip-food101-zero-shot) |

## Labels
This model supports 101 food categories including:
[
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap",
    "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake",
    "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon",
    "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
    "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels",
    "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake",
    "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
    "tiramisu", "tuna_tartare", "waffles"
]


## Dataset Info
| Split       | Number of Samples |
|-------------|-------------------|
| Training    | 5000              |
| Validation  | 1000              |
| Test        | 1000              |

(Subset manually created from original 75k image dataset, working with the whole 75k or 101k images was not possible due to long training and augmentation time)

## Data Augmentation
| Transformation             | Description |
|----------------------------|-------------|
| RandomResizedCrop(224)     | Crops and resizes randomly to 224x224 |
| RandomHorizontalFlip()     | Flips images horizontally at random |
| ColorJitter()              | Adjusts brightness and contrast |
| Resize(224,224)            | Used in validation for consistent shape |
| ToTensor()                 | Converts PIL image to PyTorch tensor |

## Model Training
### Architecture:
Model based on `google/vit-base-patch16-224` with frozen layers except final classifier.

### Hyperparameters:
- Epochs: 3
- Batch Size: 8
- Optimizer: AdamW (via `Trainer` API)
- Evaluation: Per epoch on validation set

### Accuracy:
| Epoch | Val Accuracy |
|-------|---------------|
| 1     | 93.1%         |
| 2     | 96.5%         |
| 3     | 97.0%         |

### TensorBoard:
Training progress and metrics logged locally (or optionally via HuggingFace Hub).

## Zero-Shot Evaluation
Used OpenAI's `clip-vit-base-patch32` with text prompts like `"a photo of sushi"`.

| Model                     | Accuracy (on 100 val images) |
|---------------------------|------------------------------|
| Fine-tuned ViT            | 97%                          |
| Zero-Shot CLIP            | 92%                          |

## Sample Results
![image](https://github.com/user-attachments/assets/c46ef7e6-b0ee-44c5-b17d-e6d687f3159d)


## Comparison
| Model        | Pros                             | Cons                            |
|--------------|----------------------------------|----------------------------------|
| ViT Fine-Tuned | Very accurate (97%), task-optimized | Needs training time/resources |
| CLIP Zero-Shot | No training needed, general-purpose | Lower accuracy (92%)         |

## How to Run
1. Clone Space or open Notebook in HF
2. Upload a food image
3. See top-3 predictions from ViT model
4. Compare with CLIP zero-shot predictions

## Conclusion
- Transfer learning with data augmentation significantly boosts performance.
- CLIP zero-shot models are strong out-of-the-box benchmarks.
- ViT models are highly suitable for food classification tasks.

---

Contact: [Jarin Schnierl](mailto:schnijar@students.zhaw.ch) | Project @ ZHAW AI Applications
