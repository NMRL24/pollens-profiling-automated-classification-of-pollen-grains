my project files in the structure
POLLEN_PROFILING_PROJECT/
│
├── dataset/                          # Likely contains pollen images used for training/testing
│
├── flask/
│   ├── static/
│   │   └── uploads/
│   │       ├── syagrus_01.jpg       # Uploaded test image 1
│   │       └── syagrus_04.jpg       # Uploaded test image 2
│   │
│   └── templates/
│       ├── index.html               # Homepage (upload interface)
│       ├── logout.html              # Logout page (if auth used)
│       └── predict.html             # Page showing prediction result
│
├── app.py                           # Flask backend: routes + model inference logic
├── label_encoder.pkl                # Stores label-to-class mapping (used for decoding prediction)
├── model.h5                         # Trained CNN model file
└── pollen_grain_classification.ipynb  # Jupyter Notebook used to build and train the CNN
