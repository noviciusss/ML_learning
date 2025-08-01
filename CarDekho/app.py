import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="CarDekho Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .feature-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and return the original dataset for reference"""
    try:
        df = pd.read_csv('cardekho_dataset.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'], axis=1)
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Using backup data structure.")
        # Create a minimal dataset structure for the app to work
        return create_minimal_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_minimal_dataset()

def create_minimal_dataset():
    """Create a minimal dataset structure if main dataset is not available"""
    return pd.DataFrame({
        'model': ['Swift', 'i20', 'Alto', 'City', 'Verna'],
        'vehicle_age': [5, 4, 6, 3, 5],
        'km_driven': [50000, 40000, 60000, 30000, 45000],
        'seller_type': ['Individual', 'Dealer', 'Individual', 'Dealer', 'Individual'],
        'fuel_type': ['Petrol', 'Petrol', 'Petrol', 'Petrol', 'Diesel'],
        'transmission_type': ['Manual', 'Manual', 'Manual', 'Automatic', 'Manual'],
        'mileage': [18.9, 20.1, 22.0, 17.8, 23.4],
        'engine': [1197, 1197, 796, 1497, 1582],
        'max_power': [89.8, 82.0, 46.3, 117.3, 126.2],
        'seats': [5, 5, 5, 5, 5],
        'selling_price': [500000, 600000, 300000, 800000, 700000]
    })

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        with open('car_price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['preprocessor'], model_data['label_encoder']
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return None, None, None

def train_and_save_model():
    """Train the model and save it"""
    df = load_data()
    if df is None:
        return False
    
    # Data preprocessing
    df = df.drop(['car_name', 'brand'], axis=1, errors='ignore')
    
    # Prepare features and target
    X = df.drop(['selling_price'], axis=1)
    y = df['selling_price']
    
    # Apply log transformation to target
    y_log = np.log1p(y)
    
    # Encode model feature
    label_encoder = LabelEncoder()
    X['model'] = label_encoder.fit_transform(X['model'])
    
    # Define feature types
    num_features = X.select_dtypes(exclude='object').columns
    cat_features = ['seller_type', 'fuel_type', 'transmission_type']
    
    # Create preprocessor
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer([
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features)
    ], remainder='passthrough')
    
    # Fit preprocessor and transform data
    X_processed = preprocessor.fit_transform(X)
    
    # Train XGBoost model
    model = XGBRegressor(random_state=42)
    model.fit(X_processed, y_log)
    
    # Save model and preprocessor
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder
    }
    
    with open('car_price_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return True

def predict_price(model, preprocessor, label_encoder, input_data):
    """Make prediction using the trained model"""
    try:
        # Create input dataframe
        input_df = pd.DataFrame([input_data])
        
        # Encode model feature
        input_df['model'] = label_encoder.transform([input_data['model']])[0]
        
        # Make sure columns are in the right order
        expected_columns = ['model', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power', 'seats']
        input_df = input_df.reindex(columns=expected_columns)
        
        # Transform using preprocessor
        input_processed = preprocessor.transform(input_df)
        
        # Make prediction (on log scale)
        prediction_log = model.predict(input_processed)[0]
        
        # Convert back to original scale
        prediction = np.expm1(prediction_log)
        
        return prediction
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🚗 CarDekho Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict car prices using Machine Learning with XGBoost</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for model management
    st.sidebar.markdown('<h2 class="sub-header">Model Management</h2>', unsafe_allow_html=True)
    
    if st.sidebar.button("Train New Model", help="Train a fresh XGBoost model"):
        with st.spinner("Training model... Please wait."):
            if train_and_save_model():
                st.sidebar.success("Model trained and saved successfully!")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to train model")
    
    # Load model and preprocessor
    model, preprocessor, label_encoder = load_model_and_preprocessor()
    
    if model is None:
        st.warning("No trained model found. Please train a model first using the sidebar.")
        if st.button("Train Model Now"):
            with st.spinner("Training model... Please wait."):
                if train_and_save_model():
                    st.success("Model trained successfully!")
                    st.experimental_rerun()
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🔧 Car Features</h2>', unsafe_allow_html=True)
        
        # Input features in organized sections
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("**Basic Information**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                year = st.number_input(
                    "Year of Manufacture", 
                    min_value=1990, 
                    max_value=2024, 
                    value=2015,
                    help="Year the car was manufactured"
                )
                
                km_driven = st.number_input(
                    "Kilometers Driven", 
                    min_value=0, 
                    max_value=500000, 
                    value=50000,
                    help="Total kilometers driven"
                )
            
            with col_b:
                mileage = st.number_input(
                    "Mileage (kmpl)", 
                    min_value=5.0, 
                    max_value=50.0, 
                    value=15.0,
                    step=0.1,
                    help="Fuel efficiency in km per liter"
                )
                
                engine = st.number_input(
                    "Engine (CC)", 
                    min_value=500, 
                    max_value=5000, 
                    value=1200,
                    help="Engine displacement in cubic centimeters"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("**Performance & Specifications**")
            
            col_c, col_d = st.columns(2)
            with col_c:
                max_power = st.number_input(
                    "Max Power (bhp)", 
                    min_value=50.0, 
                    max_value=1000.0, 
                    value=100.0,
                    step=0.1,
                    help="Maximum power output in brake horsepower"
                )
            
            with col_d:
                seats = st.selectbox(
                    "Number of Seats", 
                    options=sorted(df['seats'].unique()),
                    index=2,
                    help="Seating capacity of the car"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("**Car Details**")
            
            col_e, col_f, col_g = st.columns(3)
            with col_e:
                model_name = st.selectbox(
                    "Model", 
                    options=sorted(df['model'].unique()),
                    help="Car model"
                )
            
            with col_f:
                seller_type = st.selectbox(
                    "Seller Type", 
                    options=df['seller_type'].unique(),
                    help="Type of seller"
                )
            
            with col_g:
                fuel_type = st.selectbox(
                    "Fuel Type", 
                    options=df['fuel_type'].unique(),
                    help="Type of fuel used"
                )
            
            transmission_type = st.selectbox(
                "Transmission Type", 
                options=df['transmission_type'].unique(),
                help="Type of transmission"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">💰 Price Prediction</h2>', unsafe_allow_html=True)
        
        # Prepare input data (convert year to vehicle_age)
        current_year = 2024  # You can update this or make it dynamic
        vehicle_age = current_year - year
        
        input_data = {
            'model': model_name,
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'seller_type': seller_type,
            'fuel_type': fuel_type,
            'transmission_type': transmission_type,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }
        
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner("Calculating price..."):
                predicted_price = predict_price(model, preprocessor, label_encoder, input_data)
                
                if predicted_price is not None:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted Price: ₹{predicted_price:,.2f}")
                    st.markdown(f"**In Lakhs: ₹{predicted_price/100000:.2f} L**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("### 📊 Price Insights")
                    
                    # Price range indication
                    if predicted_price < 300000:
                        price_category = "Budget-friendly"
                        emoji = "💚"
                    elif predicted_price < 800000:
                        price_category = "Mid-range"
                        emoji = "💛"
                    elif predicted_price < 1500000:
                        price_category = "Premium"
                        emoji = "🧡"
                    else:
                        price_category = "Luxury"
                        emoji = "❤️"
                    
                    st.markdown(f"{emoji} **Category:** {price_category}")
                    
                    # Show price in different formats
                    st.markdown("**Price Formats:**")
                    st.markdown(f"- **Rupees:** ₹{predicted_price:,.0f}")
                    st.markdown(f"- **Lakhs:** ₹{predicted_price/100000:.2f} L")
                    if predicted_price >= 1000000:
                        st.markdown(f"- **Crores:** ₹{predicted_price/10000000:.2f} Cr")
        
        # Model information
        st.markdown("---")
        st.markdown("### 🤖 Model Info")
        st.markdown("**Algorithm:** XGBoost Regressor")
        st.markdown("**Features:** 10 input features")
        st.markdown("**Target:** Log-transformed selling price")
    
    # Dataset information
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars", len(df))
    with col2:
        st.metric("Unique Models", df['model'].nunique())
    with col3:
        st.metric("Price Range", f"₹{df['selling_price'].min():,.0f} - ₹{df['selling_price'].max():,.0f}")
    with col4:
        st.metric("Avg Price", f"₹{df['selling_price'].mean():,.0f}")
    
    # Show sample data
    if st.checkbox("Show Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
