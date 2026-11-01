# 💊 RxLedger — Smart Pharmacy Management System

**RxLedger** is a web-based pharmacy management system that helps pharmacists manage medicine inventory, monitor stock levels, track sales, and analyze medicine demand using Machine Learning.

The project combines a simple web interface with data-driven demand prediction to support better inventory management and reduce the risk of overstocking or running out of essential medicines.

## 🚀 Features

* **📊 Dashboard:** View pharmacy information and important inventory statistics.
* **💊 Medicine Inventory:** Manage medicine details and stock information.
* **🛒 Sales Tracking:** Record and monitor medicine sales.
* **📉 Demand Prediction:** Use Machine Learning to estimate future medicine demand.
* **🤖 Model Comparison:** Compare Linear Regression, Random Forest, and Gradient Boosting models.
* **📈 Performance Analysis:** Evaluate models using MAE, RMSE, R², and WAPE.
* **💾 Local Data Storage:** Store application data in the browser using local storage.

## 🛠️ Technologies Used

| Technology   | Purpose                                          |
| ------------ | ------------------------------------------------ |
| HTML         | Structures the web application                   |
| CSS          | Designs the user interface                       |
| JavaScript   | Adds interactivity and application functionality |
| Python       | Implements the Machine Learning workflow         |
| Pandas       | Handles and processes sales data                 |
| Scikit-learn | Trains and evaluates ML models                   |
| CSV          | Stores the medicine sales dataset                |
| JSON         | Stores model results and prediction information  |

## 🧠 Machine Learning Models

RxLedger compares three regression algorithms to identify a suitable model for medicine demand prediction.

1. **Linear Regression:** Establishes a baseline by learning a linear relationship between input features and demand.
2. **Random Forest Regression:** Combines multiple decision trees to predict demand.
3. **Gradient Boosting Regression:** Builds an ensemble of decision trees sequentially to improve prediction accuracy.

### Evaluation Metrics

* **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
* **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
* **R² Score:** Indicates how well a model explains variation in the target values.
* **WAPE (Weighted Absolute Percentage Error):** Measures total absolute prediction error relative to total actual demand.

The models are compared using these metrics to identify the best-performing model on the evaluated dataset.

## 📁 Project Structure

```text
RxLedger/
├── index.html          # Main web application
├── train_model.py      # Machine Learning training and evaluation
├── pharma_sales.csv    # Pharmacy sales dataset
├── ml_results.json     # Model evaluation results
├── model_report.png    # Model comparison visualization
└── README.md           # Project documentation
```

## ⚙️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/rxledger-pharmacy.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the Project

```bash
cd rxledger-pharmacy
```

### 3. Run the Website

Open `index.html` in your web browser.

Alternatively, open the project in Visual Studio Code and use the Live Server extension.

### 4. Run the Machine Learning Script

Ensure Python and the required libraries are installed.

```bash
python -m pip install pandas scikit-learn matplotlib
```

Then execute:

```bash
python train_model.py
```

The script trains and evaluates the models and generates the associated result files.

## 🌐 Deployment

The web interface can be deployed using **GitHub Pages**.

1. Upload the project files to a GitHub repository.
2. Ensure the main HTML file is named `index.html`.
3. Navigate to **Settings → Pages**.
4. Select the `main` branch and the root folder.
5. Save the settings and open the published website URL.

**Note:** GitHub Pages hosts static websites. It does not execute Python scripts on the server.

## 🔮 Future Enhancements

* Integration with a live Python backend for on-demand predictions.
* Database integration for centralized and persistent inventory records.
* Automated low-stock and medicine-expiry notifications.
* Improved demand forecasting using additional historical sales data.
* User authentication and role-based access.
* Interactive sales analytics and inventory reports.

## 🎯 Project Objective

The primary objective of RxLedger is to combine pharmacy inventory management with Machine Learning-based demand prediction to help pharmacists make informed stock management decisions.

## 📌 Disclaimer

This project is developed for educational purposes. Demand predictions are estimates based on the available dataset and should not replace professional judgment in medicine procurement or inventory planning.

---

**RxLedger — Smarter Inventory. Better Planning.**
