# Salary Prediction Project

## Overview
This project is a Flask-based web application designed to analyze and predict salaries based on various features and data visualizations. It allows users to upload datasets, visualize data trends, and perform salary predictions using machine learning models. The app also includes a module for association rule mining to identify patterns in the data.

---

## Features
- **Dataset Upload**: Users can upload datasets and preprocess them for analysis.
- **Data Visualization**: Interactive graphs such as bar charts, pie charts, scatter plots, and more.
- **Salary Prediction**: Predict salaries based on selected features using trained regression models.
- **Case Study Module**: Analyze key trends and relationships in the dataset.
- **Association Rules**: Discover patterns and insights using association rule mining.
- **User-Friendly Interface**: Intuitive navigation and a responsive dashboard.

---

## File Structure
```
project_root/
├── app.py                  # Main Flask application
├── templates/              # HTML templates for UI
│   ├── base.html
│   ├── home.html
│   ├── choose_option.html
│   ├── casestudy.html
│   ├── prediction_input.html
│   ├── prediction_result.html
├── static/                 # Static assets (CSS, images, etc.)
├── models/                 # Trained models and scripts
├── data/                   # Sample datasets
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/salary-prediction.git
   cd salary-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Usage
1. Upload a dataset from the **Home** page.
2. Navigate to the **Case Study** section to explore data trends using graphs.
3. Use the **Prediction** module to input features and predict salaries.
4. Explore **Association Rules** for additional insights.

---

## Challenges Faced
- Handling diverse datasets and ensuring compatibility.
- Achieving model accuracy through feature selection and hyperparameter tuning.
- Designing an intuitive and interactive UI for both technical and non-technical users.
- Integrating multiple functionalities into a cohesive application.

---

## Technologies Used
- **Frontend**: HTML, CSS, Bootstrap
- **Backend**: Flask, Python
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn

---

## Future Enhancements
- Add more advanced machine learning models for better prediction accuracy.
- Implement user authentication for personalized datasets and predictions.
- Enhance visualization options with advanced filters.


---

## Contributors
- M.Rahul
- B.Sai Punith
---

## Author
[Galam Sai Rohith](https://github.com/sairohith24816)
