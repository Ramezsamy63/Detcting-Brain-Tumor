from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved h5 model
model = load_model('./Brain Tumors x 2.h5')

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

results = [
  {
      "type": "No Tumor",
      "result": "Based on the MRI scan analysis, no tumor was detected. The brain tissue appears normal.",
      "description": "Magnetic Resonance Imaging (MRI) is a non-invasive imaging technology that produces detailed anatomical images. Your scan shows a healthy and normal brain structure without any evidence of abnormal growths or lesions.",
      "severity": "normal",
      "diagnose": "There are no signs of abnormal growths, lesions, or tumors in the brain. The neural structures appear to be within healthy parameters.",
      "solution": "No treatment is necessary. Routine check-ups or MRI scans can be continued based on your physician's advice."
  },
  {
      "type": "Meningioma",
      "result": "The MRI scan indicates a well-defined mass suggestive of a meningioma, which is typically a benign tumor.",
      "description": "MRI imaging shows a slow-growing mass attached to the meninges (the layers surrounding the brain). The borders are clear, and there are no signs of invasion into surrounding tissue, suggesting a benign nature.",
      "severity": "benign",
      "diagnose": "Meningiomas are usually non-cancerous tumors that develop in the meninges. They may not cause symptoms unless they grow large enough to affect brain function.",
      "solution": "Observation through regular imaging may be sufficient. In some cases, surgical removal or radiation therapy may be recommended if the tumor causes symptoms or continues to grow."
  },
  {
      "type": "Glioma",
      "result": "The MRI analysis has detected a mass with irregular borders consistent with a glioma. Further assessment is strongly recommended.",
      "description": "MRI scans show a lesion within the brain parenchyma that demonstrates characteristics associated with gliomas, such as infiltration into surrounding tissue and heterogeneous appearance.",
      "severity": "malignant",
      "diagnose": "Gliomas are a type of tumor that arises from glial cells in the brain or spine. Some gliomas are low-grade and slow-growing, but others can be aggressive and malignant.",
      "solution": "A biopsy is usually required to determine the tumor grade. Treatment may involve surgery, radiation, and/or chemotherapy depending on the type and stage."
  },
  {
      "type": "Pituitary",
      "result": "The MRI scan reveals a small mass located in the region of the pituitary gland, which is likely a pituitary adenoma.",
      "description": "MRI imaging highlights an abnormality in the sella turcica area where the pituitary gland resides. The lesion appears localized and non-invasive, characteristics typical of pituitary adenomas.",
      "severity": "benign",
      "diagnose": "Pituitary tumors are usually benign and can affect hormone production. They are often discovered incidentally or when symptoms like vision problems or hormonal imbalances arise.",
      "solution": "Treatment may include observation, medication to regulate hormone levels, or surgical removal, especially if the tumor is pressing on the optic nerves or causing hormonal disruptions."
  }
]


@app.post("/predict")
async def predict_image(img: UploadFile = File(...)):
    # Read image
    contents = await img.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image
    image = image.resize((224, 224))  # Adjust size according to your model
    image_array = np.array(image)
 
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(image_array)

    prediction = classes[np.argmax(prediction)]
    
    # Find matching result based on prediction type
    result_data = next((item for item in results if item["type"] == prediction), None)
    
    return {
        "filename": img.filename,
        "type": result_data["type"],
        "result": result_data["result"],
        "description": result_data["description"],
        "severity": result_data["severity"],
        "diagnose": result_data["diagnose"],
        "solution": result_data["solution"]
    }



import uvicorn
uvicorn.run(app, host="0.0.0.0", port=15703)