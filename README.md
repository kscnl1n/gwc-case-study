# GWC Case Study

This repository contains two notebook-based analyses focused on gender representation in technology:

- `YOLO_ads.ipynb`: an image pipeline that scrapes computer-related ad images, runs detection/classification, and produces gender-oriented summaries.
- `BLS_analysis.ipynb`: a labor statistics workflow analyzing gender participation trends in tech occupations over time.

## Project Goals

- Build a repeatable computer-vision workflow for ad imagery that analyzes the impact of representation in advertising -- specifically in CS courses.
- Quantify apparent gender representation in scraped visual media.
- Compare those findings with longer-term workforce trends from BLS data.

## Repository Contents

- `YOLO_ads.ipynb` - end-to-end ad image pipeline:
  - image scraping/collection
  - dataset preparation
  - YOLO fine-tuning (classification)
  - YOLO inference (person detection + gender classification)
  - aggregation and visualization
- `BLS_analysis.ipynb` - data cleaning, normalization, analysis, visualization, and statistical testing on BLS employment data.

## YOLO Pipeline (Detailed)

The `YOLO_ads.ipynb` notebook is organized into a staged pipeline:

1. **Scrape and collect ad images**
   - Uses search/scraping logic (including SerpAPI) to gather computer-ad images.
   - Applies deduplication and basic metadata handling.
   - Builds a working image corpus for downstream inference.

2. **Acquire and prepare labeled gender dataset**
   - Downloads a labeled gender dataset via Kaggle API token.
   - Organizes files into train/validation splits suitable for YOLO classification training.
   - Ensures class names and folder structure align with expected YOLO format.

3. **Preprocess data for YOLO classification**
   - Normalizes input images and class mapping.
   - Validates dataset completeness and split integrity.
   - Produces a clean training-ready dataset for transfer learning.

4. **Fine-tune YOLO classification model**
   - Starts from pretrained YOLO weights and fine-tunes on the gender-labeled dataset.
   - Trains a classifier intended to distinguish man/woman on cropped person regions.
   - Tracks training metrics and preserves best-performing weights.

5. **Run detection + classification on ads**
   - Executes YOLO person detection on each ad image.
   - Crops detected person regions and runs gender classification model on each crop.
   - Aggregates multi-person predictions to a per-image code:
     - `1`: man present only
     - `2`: woman present, or both man and woman present
     - `3`: no person detected
     - `4`: unclear / low confidence

6. **Aggregate results and visualize**
   - Produces a final dataset with image-level categories.
   - Computes category proportions and plots bar/stacked visual summaries.
   - Prepares outputs for interpretation alongside labor-market trends.

## How the YOLO Stages Work Together

- **Detection first, then classification:** person detection isolates candidate regions so gender classification runs on relevant crops instead of full images.
- **Transfer learning:** pretrained YOLO representations are adapted to the project-specific gender task, reducing data requirements versus training from scratch.
- **Decision rules for edge cases:** no-detection and low-confidence outcomes are explicitly coded (`3` and `4`) to prevent forced overconfident labels.
- **Image-level aggregation:** multiple detections are consolidated into a single ad-level category for easier downstream analysis.

## Typical Prerequisites

Because this repository is notebook-first, install dependencies in your Python environment before running:

- Python 3.9+
- Jupyter
- Ultralytics YOLO
- OpenCV / PIL
- Pandas, NumPy, Matplotlib, Seaborn
- Kaggle API credentials (for the gender dataset step)
- SerpAPI key (for scraping/search steps where used)

> Note: exact package versions are managed in-notebook; review notebook import/install cells before execution.

## Suggested Run Order

1. Run `YOLO_ads.ipynb` to build image-based representation outputs.
2. Run `BLS_analysis.ipynb` to generate labor statistics trend analysis.
3. Compare patterns between ad imagery and workforce participation metrics.

## Further Reading

### YOLO and Ultralytics

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [Ultralytics Classification Mode](https://docs.ultralytics.com/tasks/classify/)
- [Ultralytics Detection Mode](https://docs.ultralytics.com/tasks/detect/)
- [YOLOv8 Paper (arXiv index page)](https://arxiv.org/abs/2402.13616)

### Computer Vision Pipeline Concepts

- [Transfer Learning (CS231n notes)](https://cs231n.github.io/transfer-learning/)
- [Object Detection Overview (Papers With Code)](https://paperswithcode.com/task/object-detection)
- [Image Classification Overview (Papers With Code)](https://paperswithcode.com/task/image-classification)

### Responsible AI and Bias in Vision

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Gender Shades: Intersectional Accuracy Disparities](https://proceedings.mlr.press/v81/buolamwini18a.html)

## Notes and Caveats

- Labels inferred from appearance are inherently noisy and can encode bias. Try adding additional context outside of the BLS dataset or adding additional epochs to the training time to reduce this.
- Detection and classification confidence thresholds strongly impact category distributions.
- Any interpretation should be treated as descriptive of model outputs, not ground truth identity.
