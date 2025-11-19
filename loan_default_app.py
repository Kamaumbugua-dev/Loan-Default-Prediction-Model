import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import math

# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction Model",
    page_icon="🏦",
    layout="wide"
)

class LoanDefaultModel:
    def __init__(self):
        self.models = None
        self.data = None
        self.column_mapping = {}
        
    def detect_column_types(self, df):
        """Automatically detect which columns correspond to expected features"""
        detected_columns = {}
        available_columns = df.columns.tolist()
        
        # Common patterns for each feature type
        feature_patterns = {
            'income': ['income', 'salary', 'annual', 'wage', 'earnings', 'revenue'],
            'loansoutstanding': ['loan', 'outstanding', 'current', 'existing', 'number', 'count'],
            'loanamount': ['amount', 'loan_amount', 'principal', 'requested', 'value'],
            'employmentlength': ['employment', 'tenure', 'experience', 'years', 'length', 'duration'],
            'creditscore': ['credit', 'score', 'fico', 'rating', 'credit_score'],
            'default': ['default', 'target', 'label', 'y', 'class', 'failed', 'delinquent']
        }
        
        # Convert all to lowercase for matching
        available_lower = [str(col).lower() for col in available_columns]
        
        for feature_type, patterns in feature_patterns.items():
            for i, col in enumerate(available_lower):
                for pattern in patterns:
                    if pattern in col:
                        detected_columns[feature_type] = available_columns[i]
                        break
                if feature_type in detected_columns:
                    break
        
        return detected_columns
    
    def validate_and_prepare_data(self, df, column_mapping):
        """Prepare data using detected column mapping"""
        missing_columns = []
        required_columns = ['income', 'loansoutstanding', 'loanamount', 'employmentlength', 'creditscore', 'default']
        
        # Check which required columns we found
        for col in required_columns:
            if col not in column_mapping:
                missing_columns.append(col)
        
        # If we're missing critical columns, return None
        if len(missing_columns) > 2:  # Allow some flexibility
            return None, missing_columns
        
        # Create a new dataframe with standardized column names
        prepared_data = {}
        for expected_col, original_col in column_mapping.items():
            if expected_col in required_columns:
                prepared_data[expected_col] = df[original_col]
        
        # If default column not found, create a dummy one (for prediction only mode)
        if 'default' not in prepared_data:
            prepared_data['default'] = [0] * len(df)
            st.warning("No default column found. Using dummy values for model training.")
        
        prepared_df = pd.DataFrame(prepared_data)
        
        # Clean and convert data
        for col in prepared_df.columns:
            # Convert to numeric, coercing errors to NaN
            prepared_df[col] = pd.to_numeric(prepared_df[col], errors='coerce')
            
            # Fill NaN values
            if prepared_df[col].isna().any():
                if col == 'default':
                    fill_value = prepared_df[col].mode()[0] if not prepared_df[col].mode().empty else 0
                else:
                    fill_value = prepared_df[col].median()
                prepared_df[col] = prepared_df[col].fillna(fill_value)
        
        return prepared_df, missing_columns
    
    def sigmoid(self, z):
        """Numerically stable sigmoid function"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def train_logistic_regression_fast(self, X, y, learning_rate=0.1, iterations=50):
        """Optimized logistic regression using numpy"""
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0
        
        for _ in range(iterations):
            # Vectorized forward pass
            z = np.dot(X, weights) + bias
            predictions = self.sigmoid(z)
            
            # Vectorized backward pass
            errors = predictions - y
            dw = np.dot(X.T, errors) / m
            db = np.sum(errors) / m
            
            # Update parameters
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        return {'weights': weights.tolist(), 'bias': bias}
    
    def build_simple_tree(self, data, max_depth=2):
        """Simplified decision tree for speed"""
        if len(data) == 0 or max_depth == 0:
            default_prob = np.mean([row[-1] for row in data]) if data else 0.5
            return {'leaf': True, 'probability': default_prob}
        
        X = np.array([row[:-1] for row in data])
        y = np.array([row[-1] for row in data])
        
        # Find best simple split
        feature_variances = np.var(X, axis=0)
        best_feature = np.argmax(feature_variances)
        best_value = np.median(X[:, best_feature])
        
        left_mask = X[:, best_feature] < best_value
        right_mask = ~left_mask
        
        left_data = [data[i] for i in range(len(data)) if left_mask[i]]
        right_data = [data[i] for i in range(len(data)) if right_mask[i]]
        
        if len(left_data) == 0 or len(right_data) == 0:
            default_prob = np.mean(y)
            return {'leaf': True, 'probability': default_prob}
        
        return {
            'leaf': False,
            'feature': best_feature,
            'value': best_value,
            'left': self.build_simple_tree(left_data, max_depth - 1),
            'right': self.build_simple_tree(right_data, max_depth - 1)
        }
    
    def predict_tree_fast(self, node, row):
        if node['leaf']:
            return node['probability']
        
        if row[node['feature']] < node['value']:
            return self.predict_tree_fast(node['left'], row)
        else:
            return self.predict_tree_fast(node['right'], row)
    
    def normalize_data_fast(self, data):
        """Fast normalization using numpy"""
        if not data:
            return [], {}
            
        features = [col for col in data[0].keys() if col != 'default']
        X = np.array([[row[feature] for feature in features] for row in data])
        
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0) + 1e-8
        
        stats = {}
        for i, feature in enumerate(features):
            stats[feature] = {'mean': means[i], 'std': stds[i]}
        
        normalized = []
        for row in data:
            new_row = row.copy()
            for i, feature in enumerate(features):
                new_row[feature] = (row[feature] - means[i]) / stds[i]
            normalized.append(new_row)
        
        return normalized, stats
    
    def apply_business_rules(self, input_data, model_prediction):
        """Apply common-sense business rules to adjust predictions"""
        income = input_data.get('income', 0)
        loan_amount = input_data.get('loanamount', 0)
        credit_score = input_data.get('creditscore', 0)
        employment_length = input_data.get('employmentlength', 0)
        loans_outstanding = input_data.get('loansoutstanding', 0)
        
        base_prob = model_prediction['avg_prob']
        adjusted_prob = base_prob
        
        # Rule 1: Debt-to-Income Ratio
        if income > 0:
            dti = (loan_amount + loans_outstanding * 10000) / income  # Simplified DTI
            if dti < 0.1:  # Very low DTI
                adjusted_prob *= 0.3
            elif dti < 0.3:  # Low DTI
                adjusted_prob *= 0.7
            elif dti > 0.8:  # High DTI
                adjusted_prob *= 1.5
        
        # Rule 2: Income level adjustments
        if income > 200000:  # High income
            adjusted_prob *= 0.5
        elif income > 100000:  # Good income
            adjusted_prob *= 0.7
        
        # Rule 3: Credit score adjustments
        if credit_score > 750:  # Excellent credit
            adjusted_prob *= 0.4
        elif credit_score > 700:  # Good credit
            adjusted_prob *= 0.7
        elif credit_score < 600:  # Poor credit
            adjusted_prob *= 1.5
        
        # Rule 4: Employment stability
        if employment_length > 10:  # Long employment
            adjusted_prob *= 0.8
        
        # Rule 5: Loan amount relative to income
        if income > 0 and loan_amount / income < 0.1:  # Small loan relative to income
            adjusted_prob *= 0.5
        
        # Ensure probability stays within reasonable bounds
        adjusted_prob = max(0.01, min(0.95, adjusted_prob))
        
        # Update the prediction with adjusted values
        adjustment_factor = adjusted_prob / base_prob if base_prob > 0 else 1.0
        model_prediction['avg_prob'] = adjusted_prob
        model_prediction['logistic_prob'] *= adjustment_factor
        model_prediction['tree_prob'] *= adjustment_factor
        model_prediction['expected_loss'] = adjusted_prob * 0.9 * loan_amount
        
        return model_prediction
    
    def train_models_fast(self, data):
        """Optimized model training"""
        if not data:
            st.error("No data available for training")
            return
            
        features = [col for col in data[0].keys() if col != 'default']
        
        if len(data) < 2:
            st.error("Not enough data for training. Need at least 2 samples.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Normalizing data...")
        normalized_data, stats = self.normalize_data_fast(data)
        progress_bar.progress(25)
        
        # Prepare data for models
        X_array = np.array([[row[feature] for feature in features] for row in normalized_data])
        y_array = np.array([row.get('default', 0) for row in data])
        
        status_text.text("Training Logistic Regression...")
        logistic_model = self.train_logistic_regression_fast(X_array, y_array, 0.1, 50)
        progress_bar.progress(60)
        
        status_text.text("Training Decision Tree...")
        tree_data = [[row[feature] for feature in features] + [row.get('default', 0)] for row in data]
        decision_tree = self.build_simple_tree(tree_data, 2)
        progress_bar.progress(90)
        
        # Calculate metrics
        status_text.text("Calculating metrics...")
        logistic_predictions = self.sigmoid(np.dot(X_array, logistic_model['weights']) + logistic_model['bias'])
        logistic_accuracy = np.mean((logistic_predictions > 0.5) == y_array)
        
        tree_predictions = [self.predict_tree_fast(decision_tree, row[:-1]) for row in tree_data]
        tree_accuracy = np.mean((np.array(tree_predictions) > 0.5) == y_array)
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        self.models = {
            'logistic': logistic_model,
            'tree': decision_tree,
            'stats': stats,
            'features': features,
            'metrics': {
                'logistic_accuracy': logistic_accuracy,
                'tree_accuracy': tree_accuracy,
                'avg_defaulter_prob': np.mean(logistic_predictions[y_array == 1]) if np.sum(y_array == 1) > 0 else 0,
                'avg_non_defaulter_prob': np.mean(logistic_predictions[y_array == 0]) if np.sum(y_array == 0) > 0 else 0
            }
        }

    def predict(self, form_data):
        if not self.models:
            return None
        
        try:
            input_data = {}
            for key, value in form_data.items():
                if value:  # Only process non-empty values
                    input_data[key] = float(value)
        except (ValueError, TypeError):
            st.error("Please enter valid numeric values in all fields")
            return None
        
        # Check if we have all required features
        missing_features = [feature for feature in self.models['features'] if feature not in input_data]
        if missing_features:
            st.error(f"Missing input for: {', '.join(missing_features)}")
            return None
        
        # Normalize input
        normalized_input = input_data.copy()
        for feature in self.models['features']:
            if feature in self.models['stats']:
                stats = self.models['stats'][feature]
                normalized_input[feature] = (input_data[feature] - stats['mean']) / stats['std']
        
        # Logistic regression prediction
        X = [normalized_input[feature] for feature in self.models['features']]
        z = sum(val * self.models['logistic']['weights'][i] for i, val in enumerate(X)) + self.models['logistic']['bias']
        logistic_prob = self.sigmoid(z)
        
        # Decision tree prediction
        tree_input = [input_data[feature] for feature in self.models['features']]
        tree_prob = self.predict_tree_fast(self.models['tree'], tree_input)
        
        # Average probability
        avg_prob = (logistic_prob + tree_prob) / 2
        
        # Calculate expected loss
        recovery_rate = 0.10
        loss_given_default = 1 - recovery_rate
        loan_amount = input_data.get('loanamount', 10000)
        expected_loss = avg_prob * loss_given_default * loan_amount
        
        base_prediction = {
            'logistic_prob': logistic_prob,
            'tree_prob': tree_prob,
            'avg_prob': avg_prob,
            'expected_loss': expected_loss,
            'loan_amount': loan_amount
        }
        
        # Apply business rules for common-sense adjustments
        final_prediction = self.apply_business_rules(input_data, base_prediction)
        
        return final_prediction

def main():
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = LoanDefaultModel()
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'income': 75000.0, 'loansoutstanding': 1.0, 
            'loanamount': 15000.0, 'employmentlength': 5.0, 'creditscore': 720.0
        }
    if 'uploaded_file_processed' not in st.session_state:
        st.session_state.uploaded_file_processed = None
    
    # Header
    st.title("Loan Default Prediction Model")
    st.subheader("Predict probability of default and expected loss for personal loans")
    
    # Data Upload Section
    st.header("Step 1: Upload Training Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload any CSV file with financial data. The app will automatically detect relevant columns.",
            key="file_uploader"
        )
    
    with col2:
        if st.button("Use Sample Data", use_container_width=True):
            # More realistic sample data with better distributions
            sample_data = """customer_id,monthly_income,current_loans,requested_amount,employment_years,credit_rating,loan_status
1,6250,1,15000,8,720,0
2,3750,2,25000,3,620,1
3,10000,1,10000,15,780,0
4,2917,3,30000,2,580,1
5,7917,1,20000,10,740,0
6,12500,0,50000,12,800,0
7,4167,4,15000,1,550,1
8,9167,1,25000,7,690,0
9,2083,5,10000,0,500,1
10,11250,0,30000,10,750,0
11,5417,2,18000,4,650,0
12,3333,3,22000,2,590,1
13,14583,0,80000,20,820,0
14,4583,2,12000,3,610,1
15,10417,1,35000,8,730,0"""
            
            try:
                df = pd.read_csv(StringIO(sample_data))
                
                # Detect columns automatically
                column_mapping = st.session_state.model.detect_column_types(df)
                st.session_state.model.column_mapping = column_mapping
                
                # Show detected mapping
                st.write("Detected Columns:")
                for expected, actual in column_mapping.items():
                    st.write(f"- {expected}: {actual}")
                
                # Validate and prepare data
                prepared_df, missing_cols = st.session_state.model.validate_and_prepare_data(df, column_mapping)
                
                if prepared_df is not None:
                    data_dict = prepared_df.to_dict('records')
                    st.session_state.model.data = data_dict
                    
                    # Train models
                    st.session_state.model.train_models_fast(data_dict)
                    
                    st.session_state.data_uploaded = True
                    st.success("Sample data loaded and models trained successfully!")
                    st.rerun()
                else:
                    st.error(f"Could not prepare sample data. Missing columns: {missing_cols}")
                    st.info("The app will still work for predictions using default values.")
                    
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Process uploaded file
    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file_processed:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file_processed = uploaded_file
            
            # Display file info immediately
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            with st.expander("View Uploaded Data", expanded=True):
                st.write("Data Preview:")
                st.dataframe(df.head())
                st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                st.write("Columns:", df.columns.tolist())
            
            # Detect columns automatically
            with st.spinner("Detecting columns..."):
                column_mapping = st.session_state.model.detect_column_types(df)
                st.session_state.model.column_mapping = column_mapping
            
            # Show detected mapping
            st.subheader("Detected Column Mapping")
            mapping_data = []
            required_columns = ['income', 'loansoutstanding', 'loanamount', 'employmentlength', 'creditscore', 'default']
            
            for col in required_columns:
                status = "✅ Found" if col in column_mapping else "❌ Not found"
                original_col = column_mapping.get(col, "N/A")
                mapping_data.append({"Feature": col, "Status": status, "Mapped Column": original_col})
            
            st.table(pd.DataFrame(mapping_data))
            
            # Validate and prepare data
            with st.spinner("Preparing data..."):
                prepared_df, missing_cols = st.session_state.model.validate_and_prepare_data(df, column_mapping)
            
            if prepared_df is not None:
                data_dict = prepared_df.to_dict('records')
                st.session_state.model.data = data_dict
                
                # Train models
                with st.spinner("Training machine learning models..."):
                    st.session_state.model.train_models_fast(st.session_state.model.data)
                
                st.session_state.data_uploaded = True
                st.success("Data uploaded and models trained successfully!")
                    
            else:
                st.warning(f"Some columns not detected: {missing_cols}")
                st.info("You can still make predictions, but model accuracy may be affected.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check that your file is a valid CSV file and try again.")
    
    # Model Performance Section
    if st.session_state.data_uploaded and st.session_state.model.models:
        st.header("Model Performance")
        
        metrics = st.session_state.model.models['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Logistic Regression Accuracy",
                f"{metrics['logistic_accuracy'] * 100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Decision Tree Accuracy",
                f"{metrics['tree_accuracy'] * 100:.1f}%"
            )
        
        # Data quality assessment
        st.subheader("Data Quality Assessment")
        if metrics['logistic_accuracy'] > 0.8:
            st.success("Excellent data quality - strong patterns detected")
        elif metrics['logistic_accuracy'] > 0.6:
            st.info("Good data quality - reasonable patterns detected")
        else:
            st.warning("Lower data quality - patterns may be weak")
    
    # Prediction Form
    st.header("Step 2: Make Predictions")
    
    if st.session_state.data_uploaded and st.session_state.model.models:
        st.subheader("Enter Borrower Information")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                income = st.number_input(
                    "Annual Income ($)",
                    min_value=0.0,
                    value=float(st.session_state.form_data['income']),
                    step=1000.0,
                    help="Total annual income of the borrower",
                    key="income_input"
                )
                loan_amount = st.number_input(
                    "Loan Amount ($)",
                    min_value=0.0,
                    value=float(st.session_state.form_data['loanamount']),
                    step=1000.0,
                    help="Amount of loan being requested",
                    key="loanamount_input"
                )
                credit_score = st.number_input(
                    "Credit Score",
                    min_value=300.0,
                    max_value=850.0,
                    value=float(st.session_state.form_data['creditscore']),
                    step=10.0,
                    help="Credit score (typically 300-850)",
                    key="creditscore_input"
                )
            
            with col2:
                loans_outstanding = st.number_input(
                    "Current Loans Outstanding",
                    min_value=0.0,
                    value=float(st.session_state.form_data['loansoutstanding']),
                    step=1.0,
                    help="Number of existing loans",
                    key="loansoutstanding_input"
                )
                employment_length = st.number_input(
                    "Employment Length (years)",
                    min_value=0.0,
                    value=float(st.session_state.form_data['employmentlength']),
                    step=1.0,
                    help="Years at current employment",
                    key="employmentlength_input"
                )
            
            submitted = st.form_submit_button(
                "Calculate Default Probability and Expected Loss",
                use_container_width=True
            )
            
            if submitted:
                # Update form data with current values
                st.session_state.form_data = {
                    'income': income,
                    'loansoutstanding': loans_outstanding,
                    'loanamount': loan_amount,
                    'employmentlength': employment_length,
                    'creditscore': credit_score
                }
                
                # Convert to strings for prediction function (which converts back to float)
                form_data_str = {
                    'income': str(income),
                    'loansoutstanding': str(loans_outstanding),
                    'loanamount': str(loan_amount),
                    'employmentlength': str(employment_length),
                    'creditscore': str(credit_score)
                }
                
                try:
                    st.session_state.prediction = st.session_state.model.predict(form_data_str)
                except Exception as e:
                    st.error(f"Error calculating prediction: {e}")
    else:
        st.info("Upload data or use sample data to enable predictions")
    
    # Results Section
    if st.session_state.prediction:
        st.header("Prediction Results")
        
        pred = st.session_state.prediction
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Logistic Regression PD",
                f"{pred['logistic_prob'] * 100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Decision Tree PD",
                f"{pred['tree_prob'] * 100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Average PD",
                f"{pred['avg_prob'] * 100:.1f}%"
            )
        
        st.metric(
            "Expected Loss",
            f"${pred['expected_loss']:,.2f}",
            delta=f"on ${pred['loan_amount']:,.0f} loan"
        )
        
        # Risk Assessment with better logic
        st.subheader("Risk Assessment")
        avg_pd = pred['avg_prob']
        
        # Get current form values directly from the inputs (no session state conversion needed)
        income = st.session_state.form_data['income']
        loan_amount = st.session_state.form_data['loanamount']
        credit_score = st.session_state.form_data['creditscore']
        
        # Adjusted risk assessment that considers the actual financial situation
        if income > 0:
            loan_to_income = loan_amount / income
            
            if avg_pd < 0.05 or (income > 200000 and loan_to_income < 0.1 and credit_score > 700):
                st.success("VERY LOW RISK - Excellent candidate for approval")
                st.write("This borrower demonstrates strong financial capacity with low default probability.")
            elif avg_pd < 0.15 or (income > 100000 and loan_to_income < 0.3 and credit_score > 650):
                st.info("LOW RISK - Good candidate for approval")
                st.write("This borrower shows favorable financial indicators with acceptable risk levels.")
            elif avg_pd < 0.3:
                st.warning("MODERATE RISK - Standard approval recommended")
                st.write("This borrower presents average risk levels. Standard underwriting applies.")
            elif avg_pd < 0.5:
                st.error("HIGH RISK - Additional scrutiny required")
                st.write("Consider higher interest rates, additional collateral, or reduced loan amount.")
            else:
                st.error("VERY HIGH RISK - Caution advised")
                st.write("High probability of default. Consider rejection or require substantial collateral.")
        
        # Show key factors influencing the decision
        st.subheader("Key Decision Factors")
        factors = []
        
        if income > 200000:
            factors.append("✅ High income level")
        elif income > 100000:
            factors.append("✅ Good income level")
        else:
            factors.append("⚠️ Moderate income level")
            
        if credit_score > 750:
            factors.append("✅ Excellent credit score")
        elif credit_score > 700:
            factors.append("✅ Good credit score")
        elif credit_score > 650:
            factors.append("⚠️ Fair credit score")
        else:
            factors.append("❌ Poor credit score")
            
        if loan_amount / income < 0.1:
            factors.append("✅ Low debt-to-income ratio")
        elif loan_amount / income < 0.3:
            factors.append("⚠️ Moderate debt-to-income ratio")
        else:
            factors.append("❌ High debt-to-income ratio")
            
        for factor in factors:
            st.write(factor)
    
    # Explanation Section
    st.header("Understanding the Results")
    
    with st.expander("How to Interpret Probability of Default (PD)"):
        st.write("""
        **Probability of Default (PD)** represents the likelihood that a borrower will fail to repay their loan.
        
        **Realistic Interpretation Guide:**
        - **0-5%**: Very Low Risk - Excellent candidate (high income, strong credit)
        - **5-15%**: Low Risk - Good candidate (stable finances, good credit)  
        - **15-30%**: Moderate Risk - Standard approval (average risk profile)
        - **30-50%**: High Risk - Requires conditions (weaker financials)
        - **50%+**: Very High Risk - Strong caution (high likelihood of default)
        
        **The model considers:**
        - Machine learning predictions from your training data
        - Business rules for common-sense adjustments
        - Debt-to-income ratios and financial capacity
        - Credit history and employment stability
        """)
    
    with st.expander("Why High Income = Lower Risk"):
        st.write("""
        **Financial Logic Behind the Model:**
        
        A borrower with $5,000,000 annual income requesting a $15,000 loan represents:
        - **Extremely low debt-to-income ratio** (0.3%)
        - **Strong repayment capacity** 
        - **Minimal financial strain**
        
        **Compared to typical scenarios:**
        - Average borrower: $75,000 income, $15,000 loan = 20% DTI
        - Your scenario: $5,000,000 income, $15,000 loan = 0.3% DTI
        
        The model applies business rules to recognize these favorable conditions and adjust predictions accordingly.
        """)

if __name__ == "__main__":
    main()