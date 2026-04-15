🚗 Vehicle Fuel Efficiency Analysis & Prediction
📌 Project Overview

This project focuses on analyzing vehicle fuel efficiency using data analysis and machine learning techniques. It explores how factors such as engine size, number of cylinders, fuel type, and manufacturing year impact fuel consumption (MPG).

The project also builds a predictive model to estimate a vehicle's fuel efficiency based on its engine characteristics.

🎯 Objectives
Analyze vehicle data to identify patterns affecting fuel efficiency
Perform data cleaning and preprocessing
Visualize relationships between different variables
Categorize vehicles based on mileage performance
Build a machine learning model to predict fuel efficiency
Identify the most fuel-efficient vehicles


📂 Dataset

The dataset contains information about vehicles, including:
Make and Model
Year
Fuel Type
Engine Size
Cylinders
City MPG
Highway MPG
Combined MPG



🧹 Data Preprocessing

Removed missing values
Removed duplicate records
Selected relevant columns
Renamed columns for clarity


⚙️ Feature Engineering
New features created:
Efficiency Score → Average of City and Highway MPG
MPG Category → Low, Medium, High
Engine Category → Small, Medium, Large

📊 Exploratory Data Analysis (EDA)
The following visualizations were used:
Engine Size vs City MPG (Scatter Plot)
Fuel Type vs MPG (Box Plot)
Year vs MPG Trend (Line Plot)
Correlation Heatmap
MPG Distribution (Histogram)
Engine Category vs MPG (Bar Plot)
Cylinders vs MPG (Box Plot)

📈 Key Insights
Larger engine sizes generally result in lower fuel efficiency
Vehicles with fewer cylinders tend to have higher MPG
Fuel efficiency has improved over the years
Strong negative correlation between engine size and MPG

🤖 Machine Learning Model
Model Used: Linear Regression
Input Features: Engine Size Cylinders
Target Variable: Combined MPG

📌 Model Performance
Evaluated using: R² Score
Mean Squared Error (MSE)

🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

📁 Project Structure
├── vehicles.csv
├── cleaned_vehicle_data.csv
├── Analysis.py
├── README.md

▶️ How to Run
Install required libraries:
1. pip install pandas numpy matplotlib seaborn scikit-learn
Run the script:
2.python Analysis.py

📌 Conclusion

This project demonstrates how data analysis and machine learning can be used to understand and predict vehicle fuel efficiency. It provides meaningful insights that can help in making informed decisions when selecting vehicles.

📬 Author
Dinesh Karthik Damuluri






