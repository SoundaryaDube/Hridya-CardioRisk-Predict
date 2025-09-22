# Hridya-CardioRisk-Predict
# MIT-BIH Arrhythmia Classification Project

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline for analyzing and classifying cardiac arrhythmias using the MIT-BIH Arrhythmia Database. The system extracts meaningful features from ECG signals, handles severe class imbalance, and implements various machine learning models for accurate arrhythmia detection.

## 🚀 Key Features

- **Data Processing**: Automated extraction and preprocessing of MIT-BIH ECG records
- **Feature Engineering**: Comprehensive feature extraction including time-domain, frequency-domain, and morphological characteristics
- **Class Imbalance Handling**: Implementation of SMOTE and class weighting strategies
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, and deep learning architectures
- **Comprehensive Evaluation**: Detailed performance metrics and clinical validation
- **Visualization**: Extensive ECG signal visualization and feature analysis

## 📊 Dataset

The project uses the **MIT-BIH Arrhythmia Database** containing:
- 48 half-hour excerpts of two-channel ambulatory ECG recordings
- 109,000+ annotated beats with 15 different arrhythmia types
- Sampling rate: 360 Hz
- Lead configuration: MLII (modified limb lead II) and V1

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Google Colab environment (recommended)
- Google Drive for data storage

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd mit-bih-arrhythmia-classification

# Install required packages
pip install wfdb numpy pandas matplotlib scikit-learn seaborn tensorflow torch keras xgboost lightgbm imblearn flask websocket boto3
```

### Google Colab Setup
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Extract the dataset:
```python
import zipfile
dataset_path = "/content/drive/MyDrive/Datasets/mit-bih-arrhythmia-database-1.0.0.zip"
extract_path = "/content/ECG_Data/"
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

## 🏗️ Project Structure

```
mit-bih-arrhythmia-classification/
│
├── MIT_Arrhythmia_Dataset.ipynb    # Main Colab notebook
├── mitbih_arrhythmia_features.csv  # Extracted features (generated)
├── feature_importance_analysis.csv # Feature importance results
├── clinical_validation_report.json # Validation results
└── README.md
```

## 🔧 Usage

### 1. Data Processing
```python
processor = MITBIHProcessor(data_path="/content/ECG_Data/mit-bih-arrhythmia-database-1.0.0/")
feature_df = processor.process_all_records(max_records=5)
```
<img width="641" height="661" alt="image" src="https://github.com/user-attachments/assets/a58d0938-3959-4c7d-8342-0856f263f3fc" />

### 2. Exploratory Data Analysis
```python
processor.exploratory_data_analysis()
correlations = enhanced_analysis(feature_df)
```
<img width="503" height="214" alt="image" src="https://github.com/user-attachments/assets/bb294ce9-659b-421a-9df7-153bceeced53" />
<img width="797" height="646" alt="image" src="https://github.com/user-attachments/assets/d28eab0a-d072-4358-82dd-e93b813d9d1b" />
<img width="853" height="593" alt="image" src="https://github.com/user-attachments/assets/b14d72ac-d3e4-4155-ba05-0f6f0e9ea4c9" />
<img width="863" height="533" alt="image" src="https://github.com/user-attachments/assets/d94f5c03-80da-4ac8-b531-81ccbacf72cb" />
<img width="534" height="500" alt="image" src="https://github.com/user-attachments/assets/e8a47a6b-7fc8-497f-b2ff-806e7b2b2a34" />

### 3. Model Training & Evaluation
```python
best_model, feature_importance = advanced_ml_pipeline(feature_df)
```
<img width="966" height="251" alt="image" src="https://github.com/user-attachments/assets/c7ae9a7f-20a3-4634-bf0a-50ea00dc07b3" />
<img width="1292" height="616" alt="image" src="https://github.com/user-attachments/assets/b7a93d86-f691-4b80-a046-b46af1e52508" />

### 4. Clinical Validation
```python
results = clinical_validation_pipeline(X, y)
generate_validation_report(results, X, y)
```
ECG_CONDITIONS = {
    "Normal Sinus Rhythm": {"risk": "low", "description": "Regular rhythm with normal ECG characteristics"},
    "Atrial Fibrillation": {"risk": "high", "description": "Irregularly irregular rhythm with no discernible P waves"},
    "Atrial Flutter": {"risk": "medium", "description": "Sawtooth pattern of atrial activity at 250-350 bpm"},
    "Premature Ventricular Contraction": {"risk": "medium", "description": "Early, wide QRS complex without preceding P wave"},
    "Ventricular Tachycardia": {"risk": "high", "description": "Wide QRS tachycardia > 100 bpm"},
    "Supraventricular Tachycardia": {"risk": "medium", "description": "Narrow QRS tachycardia > 150 bpm"},
    "Sinus Bradycardia": {"risk": "low", "description": "Sinus rhythm with heart rate < 60 bpm"},
    "Sinus Tachycardia": {"risk": "low", "description": "Sinus rhythm with heart rate > 100 bpm"},
    "First-Degree AV Block": {"risk": "low", "description": "PR interval > 200 ms"},
    "Second-Degree AV Block Type 1": {"risk": "medium", "description": "Progressive PR prolongation until QRS dropped"},
    "Second-Degree AV Block Type 2": {"risk": "high", "description": "Intermittent non-conducted P waves without PR prolongation"},
    "Third-Degree AV Block": {"risk": "high", "description": "Complete dissociation between P waves and QRS complexes"},
    "Left Bundle Branch Block": {"risk": "medium", "description": "QRS > 120 ms with characteristic morphology"},
    "Right Bundle Branch Block": {"risk": "medium", "description": "QRS > 120 ms with RSR' pattern in V1"},
    "Left Ventricular Hypertrophy": {"risk": "medium", "description": "Increased voltage criteria in left precordial leads"},
    "Right Ventricular Hypertrophy": {"risk": "medium", "description": "Right axis deviation with tall R waves in V1"},
    "Acute Myocardial Infarction": {"risk": "high", "description": "ST elevation in contiguous leads"},
    "Old Myocardial Infarction": {"risk": "medium", "description": "Pathologic Q waves in contiguous leads"},
    "Ischemia": {"risk": "medium", "description": "ST depression or T wave inversion"},
    "Pericarditis": {"risk": "medium", "description": "Diffuse ST elevation with PR depression"},
    "Hyperkalemia": {"risk": "high", "description": "Tall, peaked T waves with widened QRS"},
    "Hypokalemia": {"risk": "medium", "description": "ST depression, flattened T waves, prominent U waves"},
    "Long QT Syndrome": {"risk": "high", "description": "QTc > 470 ms in men or > 480 ms in women"},
    "Brugada Syndrome": {"risk": "high", "description": "Right bundle branch block pattern with ST elevation in V1-V3"},
    "Pulmonary Embolism": {"risk": "high", "description": "S1Q3T3 pattern, right heart strain"},
    "Wolff-Parkinson-White": {"risk": "medium", "description": "Short PR interval with delta wave"},
    "Sick Sinus Syndrome": {"risk": "medium", "description": "Bradycardia-tachycardia syndrome"},
    "Pacemaker Rhythm": {"risk": "low", "description": "Pacemaker spikes with subsequent depolarization"}
}

<img width="1662" height="501" alt="image" src="https://github.com/user-attachments/assets/270407b3-4d7b-46d6-8db5-6cb5891cdb21" />
<img width="746" height="583" alt="image" src="https://github.com/user-attachments/assets/cd0f6864-4b7f-4b9e-8a28-097765c640d2" />
<img width="753" height="463" alt="image" src="https://github.com/user-attachments/assets/27dcfe38-b8b0-4248-8539-f4e412ef61f6" />

## 📈 Results

### Top Performing Features:
1. Spectral Bandwidth (correlation: 0.867)
2. Kurtosis (correlation: 0.859)
3. Skewness (correlation: 0.844)
4. R-peak Amplitude (correlation: 0.716)
5. Maximum Amplitude (correlation: 0.614)

<img width="544" height="349" alt="image" src="https://github.com/user-attachments/assets/afd93179-543e-472d-bb50-70040b29c0d6" />
<img width="562" height="654" alt="image" src="https://github.com/user-attachments/assets/094ca060-4b87-4572-b509-8aa855caec12" />
<img width="577" height="659" alt="image" src="https://github.com/user-attachments/assets/ec7ffe7c-c666-45bb-8829-4f44d51ffc53" />

### Class Distribution:
- Normal Beats: 64.97%
- Paced Beats: 34.37%
- Atrial Premature Beats: 0.38%
- Unclassifiable: 0.20%
- PVCs: 0.07%

## 🎯 Model Performance

The pipeline achieves:
- High accuracy in distinguishing normal vs abnormal beats
- Effective handling of severe class imbalance
- Comprehensive feature importance analysis
- Clinical validation with medical recommendations

<img width="1675" height="587" alt="image" src="https://github.com/user-attachments/assets/df255b5c-e717-4b10-82e1-71d575908e1f" />
<img width="874" height="421" alt="image" src="https://github.com/user-attachments/assets/bfb4fe25-3d1b-4774-a492-311f0312f105" />

## 🏥 Clinical Applications

This system can assist in:
- Automated arrhythmia screening
- Remote patient monitoring
- Early detection of cardiac abnormalities
- Clinical decision support systems

## 🔮 Future Enhancements

- Real-time ECG analysis capabilities
- Mobile application integration
- Cloud-based API deployment
- Additional arrhythmia type detection
- Integration with hospital EHR systems

## 📚 References

1. Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database." IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001)
2. Goldberger AL, et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." Circulation 101(23):e215-e220 (2000)

## 👥 Contributors

[Your Name/Organization]
- ECG Signal Processing
- Machine Learning Implementation
- Clinical Validation

## 📄 License

This project is intended for research and educational purposes. Clinical use requires proper validation and regulatory approval.

## 🤝 Acknowledgments

- MIT-BIH Arrhythmia Database providers
- PhysioNet for maintaining the database
- Open-source community for ML libraries

---

**Note**: This project is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.
