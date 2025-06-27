my project files in the structure
POLLEN_PROFILING_PROJECT/
│
├── dataset/                          
│   └── [Contains pollen grain images used for training/testing the model]
│
├── flask/
│   ├── static/
│   │   └── uploads/
│   │       ├── syagrus_01.jpg       # Sample uploaded image for prediction
│   │       └── syagrus_04.jpg       # Another uploaded image
│   │
│   └── templates/
│       ├── index.html               # Web page for uploading images
│       ├── logout.html              # Logout page (optional, if user auth is used)
│       └── predict.html             # Displays the prediction results from the model
│
├── app.py                           # Main Flask application file handling routes, model loading, and predictions
├── label_encoder.pkl                # Pickled label encoder for decoding model predictions into class names
├── model.h5                         # Trained Keras CNN model for pollen classification
└── pollen_grain_classification.ipynb  # Jupyter Notebook used for training and evaluating the CNN model

