# 💻 Laptop Price Prediction (ANN)


Deployment Link :- https://laptop-price-prediction-qzrteewafuffxbxtnzunvk.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)

## 📌 Project Overview
**TechPricer** is an enterprise-grade machine learning dashboard built to predict global market valuations for laptop hardware. Migrated from a standard regression pipeline to a **Deep Dense Neural Network (ANN)**, this platform ingests 10 distinct hardware features (Brand, CPU, RAM, GPU, etc.) and executes a forward pass to determine an accurate market price. 

The system features a monolithic, 6-tab OS environment complete with hardware topology radar charts, 5-year depreciation forecasting, and Monte Carlo market variance simulations.

## 🚀 Enterprise Features
* **🧠 Deep Learning Backend:** Powered by a TensorFlow/Keras Artificial Neural Network (ANN) utilizing ReLU activations and the Adam optimizer.
* **🕸️ Hardware Topology Analytics:** Dynamic Plotly radar charts to visualize system bottlenecks (Portability vs. Compute Power).
* **📉 Depreciation Simulator:** Predicts 5-year asset depreciation based on historical decay rates for specific brands and form factors.
* **🎲 Market Variance Engine:** Executes 100-iteration Monte Carlo simulations to model global pricing volatility.
* **💾 Secure Dossier Export:** Generates downloadable JSON and CSV artifacts containing the input tensor and prediction telemetry.

## 📁 Repository Structure

📦 Laptop-Price-Prediction

    ┣ 📜 app.py               # Main Streamlit UI (TechPricer Monolithic Build)
    ┣ 📜 ann_model.h5         # Trained Keras Neural Network (Deep Learning Weights)
    ┣ 📜 preprocessor.pkl     # Scikit-Learn Data Encoder (StandardScaler/OneHot)
    ┣ 📜 requirements.txt     # Python dependency lockfile
    ┗ 📜 README.md            # System documentation

 Gemini said
Here is the complete configuration package for your GitHub repository. I have styled the README.md to match the exact enterprise-grade architecture of the TechPricer Neural Engine we just built, making your repository look highly professional to recruiters, collaborators, or stakeholders.

1. Requirements File (requirements.txt)
Save this exactly as requirements.txt in your root directory. It explicitly locks in the dependencies required to run the TensorFlow backend and the Plotly visualizations.

Plaintext
# UI & Dashboard Framework
streamlit>=1.24.0

# Data Manipulation & Array Math
pandas>=1.5.3
numpy>=1.23.5

# Machine Learning & Deep Learning Backend
tensorflow>=2.13.0
scikit-learn>=1.2.2

# Interactive Data Visualization
plotly>=5.14.0
2. GitHub Documentation (README.md)
Save the following markdown exactly as README.md. It includes a professional structure, feature highlights, and clear setup instructions.

Markdown
# 💻 Laptop Price Prediction (TechPricer Neural Engine)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)

## 📌 Project Overview
**TechPricer** is an enterprise-grade machine learning dashboard built to predict global market valuations for laptop hardware. Migrated from a standard regression pipeline to a **Deep Dense Neural Network (ANN)**, this platform ingests 10 distinct hardware features (Brand, CPU, RAM, GPU, etc.) and executes a forward pass to determine an accurate market price. 

The system features a monolithic, 6-tab OS environment complete with hardware topology radar charts, 5-year depreciation forecasting, and Monte Carlo market variance simulations.

## 🚀 Enterprise Features
    * **🧠 Deep Learning Backend:** Powered by a TensorFlow/Keras Artificial Neural Network (ANN) utilizing ReLU activations and the Adam optimizer.
    * **🕸️ Hardware Topology Analytics:** Dynamic Plotly radar charts to visualize system bottlenecks (Portability vs. Compute Power).
    * **📉 Depreciation Simulator:** Predicts 5-year asset depreciation based on historical decay rates for specific brands and form factors.
    * **🎲 Market Variance Engine:** Executes 100-iteration Monte Carlo simulations to model global pricing volatility.
    * **💾 Secure Dossier Export:** Generates downloadable JSON and CSV artifacts containing the input tensor and prediction telemetry.

## 📁 Repository Structure

📦 Laptop-Price-Prediction

    ┣ 📜 app.py               # Main Streamlit UI (TechPricer Monolithic Build)
    ┣ 📜 ann_model.h5         # Trained Keras Neural Network (Deep Learning Weights)
    ┣ 📜 preprocessor.pkl     # Scikit-Learn Data Encoder (StandardScaler/OneHot)
    ┣ 📜 requirements.txt     # Python dependency lockfile
    ┗ 📜 README.md            # System documentation


🛠️ Installation & Setup
1. Clone the repository
   
        git clone [https://github.com/akshitgajera1013/Laptop-Price-Prediction.git](https://github.com/akshitgajera1013/Laptop-Price-Prediction.git)

cd Laptop-Price-Prediction

2. Create a Virtual Environment (Recommended)

       python -m venv venv
       source venv/bin/activate  # On Windows use: venv\Scripts\activate


3. Install Dependencies
   
       pip install -r requirements.txt

4. Execute the Valuation Engine
   
       streamlit run app.py


🧪 Model Performance Metrics

      R² Score: 0.855 (Captures 85.5% of market variance)
      
      Mean Absolute Error (MAE): €182.72
      
      Mean Squared Error (MSE): 73,548.17

⚙️ Hardware Input Tensor
The neural network requires the following 10-dimensional input vector:

    Company (Apple, Dell, HP, etc.)
    
    TypeName (Ultrabook, Gaming, Notebook, etc.)
    
    Inches (Screen Scale)
    
    ScreenResolution (Panel Type & Pixel Density)
    
    Cpu (Processor Tier)
    
    Ram (System Memory in GB)
    
    Memory (Storage Configuration & Drive Type)
    
    Gpu (Graphics Compute Unit)
    
    OpSys (Operating System)
    
    Weight (Chassis Mass in kg)
