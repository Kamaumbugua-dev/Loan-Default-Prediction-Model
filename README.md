# Loan-Default-Prediction-Model

# Loan Default Prediction Model

A comprehensive machine learning application that predicts the probability of loan default and calculates expected loss for personal loans. Built with Streamlit, this tool combines statistical modeling with business intelligence to provide realistic risk assessments.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

##  Features

### Core Functionality
- **Dual Model Approach**: Combines Logistic Regression and Decision Tree algorithms
- **Automated Data Processing**: Intelligent column detection for any financial dataset
- **Business Rule Integration**: Applies financial logic to machine learning predictions
- **Real-time Predictions**: Instant probability of default and expected loss calculations
- **Comprehensive Risk Assessment**: Multi-factor risk evaluation with clear explanations

### Technical Features
- **Data Agnostic**: Works with various CSV formats and column naming conventions
- **Numerical Stability**: Robust handling of edge cases and large value ranges
- **Progress Tracking**: Real-time progress indicators during model training
- **Error Resilience**: Graceful handling of missing or imperfect data

##  Model Architecture

### Machine Learning Models
1. **Logistic Regression**
   - Linear model for probability estimation
   - Fast training and interpretable results
   - Handles linear relationships effectively

2. **Decision Tree**
   - Non-linear model for complex patterns
   - Captures feature interactions
   - Robust to outliers

### Business Logic Layer
- **Debt-to-Income Ratio Analysis**
- **Credit Score Adjustments**
- **Income Level Considerations**
- **Employment Stability Factors**
- **Loan-to-Income Ratio Assessment**

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kamaumbugua-dev/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run loan_default_app.py
   ```

### Dependencies
The application requires the following Python packages:
```text
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
```

##  Project Structure

```
loan-default-prediction/
│
├── loan_default_app.py          # Main application file
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── examples/                    # Sample datasets
│   ├── sample_loan_data.csv
│   └── large_loan_portfolio.csv
└── screenshots/                 # Application screenshots
    ├── main_interface.png
    ├── prediction_results.png
    └── data_upload.png
```

##  Usage Guide

### Step 1: Data Preparation
Prepare your CSV file with financial data. The application automatically detects columns with common names:

**Required Features:**
- **Income**: Annual income (`income`, `salary`, `annual_income`)
- **Loans Outstanding**: Number of existing loans (`loans`, `outstanding`, `current_loans`)
- **Loan Amount**: Requested loan amount (`loan_amount`, `amount`, `principal`)
- **Employment Length**: Years at current job (`employment`, `tenure`, `experience`)
- **Credit Score**: Credit rating (`credit_score`, `fico`, `rating`)
- **Default**: Target variable (`default`, `target`, `loan_status`)

### Step 2: Upload Data
1. Launch the application: `streamlit run loan_default_app.py`
2. Upload your CSV file or use the provided sample data
3. Review the automatic column mapping
4. Monitor model training progress

### Step 3: Make Predictions
1. Enter borrower information in the prediction form:
   - Annual Income
   - Loan Amount Requested
   - Credit Score
   - Current Loans Outstanding
   - Employment Length
2. Click "Calculate Default Probability and Expected Loss"
3. Review comprehensive risk assessment

##  Output Interpretation

### Probability of Default (PD)
- **0-5%**: Very Low Risk - Excellent candidate
- **5-15%**: Low Risk - Good candidate
- **15-30%**: Moderate Risk - Standard approval
- **30-50%**: High Risk - Additional scrutiny required
- **50%+**: Very High Risk - Caution advised

### Expected Loss Calculation
```
Expected Loss = PD × LGD × Loan Amount
```
Where:
- **PD** = Probability of Default
- **LGD** = Loss Given Default (90% = 1 - 10% recovery rate)

### Risk Factors Considered
- Debt-to-Income Ratio
- Credit History Quality
- Income Stability
- Employment Duration
- Loan Size Relative to Income

##  Business Applications

### Financial Institutions
- **Credit Risk Assessment**: Automated loan approval decisions
- **Portfolio Management**: Risk-weighted asset allocation
- **Regulatory Compliance**: Basel II/III capital requirement calculations
- **Pricing Strategy**: Risk-based interest rate determination

### Individual Lenders
- **Peer-to-Peer Lending**: Investment risk evaluation
- **Small Business Loans**: Creditworthiness assessment
- **Personal Finance**: Borrowing capacity analysis

##  Customization

### Model Parameters
Adjust the following parameters in the code for your specific use case:

```python
# Training parameters
learning_rate = 0.1
iterations = 50
tree_depth = 2

# Business rule weights
income_adjustment_factor = 0.5
credit_score_adjustment = 0.4
dti_threshold = 0.3
```

### Adding New Features
1. Update `detect_column_types()` method with new feature patterns
2. Modify `validate_and_prepare_data()` for data validation
3. Extend `apply_business_rules()` with new financial logic

##  Testing & Validation

### Sample Data
The application includes built-in sample data for testing:
- 15 diverse loan applications
- Mixed risk profiles (defaults and non-defaults)
- Realistic financial ranges

### Model Validation
- Cross-validation accuracy metrics
- Business rule effectiveness testing
- Edge case handling verification

##  Performance

### Training Speed
- Small datasets (<1,000 rows): 2-5 seconds
- Medium datasets (1,000-10,000 rows): 5-15 seconds
- Large datasets (>10,000 rows): 15-30 seconds

### Accuracy Metrics
- Typical training accuracy: 70-85%
- Robust to missing data and outliers
- Consistent across different dataset sizes

##  Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Reporting Issues
Please use the [GitHub Issues](https://github.com/yourusername/loan-default-prediction/issues) page to report bugs or suggest enhancements.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for rapid web application development
- Machine learning algorithms implemented from first principles
- Financial risk modeling based on industry best practices
- Inspired by real-world credit risk assessment challenges

##  Support

For support and questions:
-  Email: stevenk710@gmail.com
- GitHub Issues: [Create an issue](https://github.com/kamaumbugua-dev/loan-default-prediction/issues)
-  Twitter: [@fahari_dubu](https://twitter.com/fahari_dubu)

##  Future Enhancements

- [ ] Additional machine learning models (Random Forest, XGBoost)
- [ ] Real-time data integration capabilities
- [ ] Advanced feature engineering
- [ ] Model explainability (SHAP values)
- [ ] API endpoint for batch processing
- [ ] Database integration for persistent storage

---

**Disclaimer**: This tool is for educational and demonstration purposes. Always consult with financial professionals for real-world lending decisions.

---


**⭐ Star this repository if you find it helpful!**

</div>
