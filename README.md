# 🌿 CropDoctor — AI-Based Crop Disease Detection

**SDG 2 — Zero Hunger** | MobileNetV2 | PyTorch | Streamlit

CropDoctor is an AI-powered crop disease detection system that identifies plant diseases from leaf images and provides suggested treatment advice. It is an educational internship project supporting UN Sustainable Development Goal 2: Zero Hunger.

## Features

* Detect supported crop disease classes from leaf images.
* Use MobileNetV2 transfer learning with PyTorch.
* Provide treatment suggestions using a local advice dictionary.
* Offer a Streamlit web interface.
* Support optional LLM-based advice when configured.
* Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Project Structure

```text
CropDoctor/
├── models/
│   ├── class_names.json
│   ├── config.json
│   └── training_history.json
├── advice.py
├── app.py
├── dataset.py
├── evaluate.py
├── predictor.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

The trained model and dataset may need to be downloaded or generated separately.

## Setup — Windows

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset and Training

Download the PlantVillage dataset and place the selected class folders in the directory expected by `dataset.py`.

Check the supported command-line arguments with:

```powershell
python train.py --help
```

Then train the model using the arguments supported by your version of `train.py`.

Evaluate the model:

```powershell
python evaluate.py --help
```

Run the evaluation command with the appropriate dataset path and options.

## Run the Application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal.

## Limitations

* The model supports only the classes on which it was trained.
* Performance on real farm images may differ from performance on controlled dataset images.
* Treatment guidance is general and should not replace professional agricultural advice.
* Confirm diagnoses with a qualified agricultural expert before applying treatments.

## License and Dataset

The repository's software license applies to your project code. The PlantVillage dataset and any pretrained model weights may have separate license terms; review those terms before redistribution.

---

*CropDoctor — an internship project supporting SDG 2: Zero Hunger.*
