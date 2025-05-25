---

# **Breast Cancer Prediction: Interpretable Machine Learning Model**

## **Project Overview**
This project aims to build an **interpretable machine learning model** to classify breast tumors as **benign** or **malignant** using the Wisconsin Breast Cancer Dataset. The focus is on achieving high accuracy (**F1 Score > 0.95**) while ensuring the model is **trustworthy and explainable** for oncologists.

---

## **Project Goals**
1. Develop a machine learning model to classify breast cancer tumors accurately.
2. Use interpretability techniques (SHAP) to explain the predictions.
3. Design a workflow that can be integrated into clinical practice.
4. Optimize precision and recall through threshold adjustment.

---

## **Dataset**
- **Source**: Wisconsin Breast Cancer Dataset from the UCI ML Repository.  
- **Attributes**:
   - **30 numerical features** describing cell nucleus properties.  
   - **Target Variable**:  
     - `M` → Malignant (1)  
     - `B` → Benign (0).  
- **Class Distribution**:
   - Malignant: 212 samples  
   - Benign: 357 samples  

---

## **Tools and Libraries**
The following libraries are used in this project:
- **Python** (3.8+)
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning modeling
- **Matplotlib & Seaborn**: Visualizations
- **SHAP**: Model interpretability
- **Streamlit**: Web-based GUI for predictions

---

## **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. **Install Dependencies**:
   Install all required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Exploratory Data Analysis**:
   Open the Jupyter notebook for EDA:
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

4. **Train the Model**:
   Run the `train_model.py` script to train the Random Forest and Voting Classifier models:
   ```bash
   python scripts/train_model.py
   ```

5. **Run the GUI**:
   Launch the Streamlit web app to input patient data and view predictions:
   ```bash
   streamlit run gui/streamlit_app.py
   ```

---

## **Key Results**
- **Model Used**: Random Forest Classifier and Voting Classifier.  
- **Performance**:
   - **F1 Score**: 0.97  
   - **Precision**: 0.98  
   - **Recall**: 0.96  

- **Interpretability**:  
   Top 3 influential features identified using SHAP:
   1. `radius_worst`  
   2. `concave_points_mean`  
   3. `area_mean`

---

## **GUI Description**
The interactive GUI allows oncologists to:
- Input patient tumor data.
- View predictions (Benign/Malignant) with confidence scores.
- Understand key features influencing predictions using SHAP insights.

---

## **Future Improvements**
- Integrate with electronic health record (EHR) systems for real-time predictions.
- Add batch predictions for multiple patients.
- Explore deep learning techniques for higher accuracy.

---

## **References**
1. Scikit-learn Documentation: https://scikit-learn.org  
2. SHAP Documentation: https://shap.readthedocs.io  
3. UCI ML Repository: Wisconsin Breast Cancer Dataset    

---
