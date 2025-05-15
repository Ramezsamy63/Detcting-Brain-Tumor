# Brain Tumor Detection

This project uses deep learning to detect brain tumors from MRI images. It consists of a trained model and an API for making predictions.

## Overview

The Brain Tumor Detection system analyzes MRI scans to identify the presence of brain tumors. The project includes:

- A trained deep learning model for tumor classification
- A web API for making predictions
- A user interface for uploading and analyzing MRI images

## Installation

### Prerequisites

- Python 3.7+
- pip package manager
- Git

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/AbdellatifOsama/DetectingBrainTumor.git
```

2. Navigate to the project directory:

```bash
cd DetectingBrainTumor
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
   
   Download the model file from [this link](https://drive.google.com/file/d/1qzByxmy-opRlauigeF0TrFj01lSDqoW2/view?usp=sharing)

5. Copy the downloaded model file to the API folder:

```bash
cp path/to/downloaded/model.h5 api/
```

## Usage

### Running the API

1. Navigate to the API directory:

```bash
cd api
```

2. Start the API server:

```bash
python app.py
```

3. The API will be available at `http://localhost:5000`

### Making Predictions

You can use the API to make predictions by sending POST requests with MRI images to the `/predict` endpoint.

## Model Information

The brain tumor detection model was trained on a dataset of MRI scans and can classify images into:
- Normal brain tissue
- Brain tumor present

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.