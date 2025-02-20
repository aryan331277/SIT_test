import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(page_title="Simple Time Series Forecast", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2a3f5f; font-size: 36px;}
    .st-bw {background-color: #000000;}
    .css-18e3th9 {padding: 2rem 5rem;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput>div>div>input {background-color: #e8f0fe;}
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    st.title("📈 Simple Time Series Forecasting")
    st.markdown("""<p style="font-size: 18px; color: #555555;">Upload your time series data or enter manually to generate forecasts.</p>""", unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Data input options
    with st.expander("📤 Data Input", expanded=True):
        input_method = st.radio("Choose data input method:", 
                              ("Upload CSV", "Manual Input"), horizontal=True)

        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your time series CSV file", 
                                           type=["csv"], help="Select your CSV file here.")
            if uploaded_file:
                try:
                    # Read CSV without specifying dtype for now
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success("Data uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")

        else:
            with st.form("manual_input_form"):
                manual_data = st.data_editor(
                    pd.DataFrame(columns=["date", "value"]),
                    num_rows="dynamic",
                    column_config={
                        "date": st.column_config.DateColumn(
                            "Date",
                            help="Select date for observation",
                            format="YYYY-MM-DD",
                        ),
                        "value": st.column_config.NumberColumn(
                            "Value",
                            help="Enter numerical value",
                        )
                    },
                    height=300
                )
                if st.form_submit_button("Save Manual Data"):
                    # Convert date column to string to ensure consistency
                    manual_data['date'] = manual_data['date'].astype(str)
                    st.session_state.df = manual_data
                    st.success("Manual data saved!")

    # Main analysis section
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(5), use_container_width=True)

        # Column selection
        cols = df.columns.tolist()
        date_col, value_col = st.columns(2)
        with date_col:
            date_column = st.selectbox("Select Date Column", options=cols)
        with value_col:
            value_column = st.selectbox("Select Value Column", options=cols)

        # Ensure date column is string type
        df[date_column] = df[date_column].astype(str)

        # Data validation and conversion
        df = df.dropna(subset=[date_column, value_column])  # Remove rows with NaN
        df = df[df[date_column].str.match(r'\d{4}-\d{2}-\d{2}')]  # Keep only well-formatted dates

        # Convert date column
        try:
            df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=[date_column])  # Drop rows with invalid dates
        except Exception as e:
            st.error(f"Error converting date column to datetime: {str(e)}")
            return

        # Convert value column to numeric if necessary
        try:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        except Exception as e:
            st.error(f"Error converting value column to numeric: {str(e)}")
            return

        # Time series visualization
        st.subheader("Time Series Overview")
        fig = px.line(df, x=date_column, y=value_column, title="Historical Data")
        st.plotly_chart(fig, use_container_width=True)

        # Simple Forecasting
        st.subheader("🔮 Forecast")
        with st.form("forecast_settings"):
            days_to_forecast = st.number_input("Days to Forecast", min_value=1, max_value=365, value=30)
            if st.form_submit_button("Predict"):
                with st.spinner("Generating forecast..."):
                    try:
                        # Prepare data for Prophet
                        prophet_df = df[[date_column, value_column]]
                        prophet_df.columns = ['ds', 'y']
                        prophet_df['ds'] = prophet_df['ds'].dt.strftime('%Y-%m-%d')  # Ensure date format

                        # Model training
                        model = Prophet()
                        model.fit(prophet_df)

                        # Future dataframe
                        future = model.make_future_dataframe(periods=days_to_forecast)
                        
                        # Generate forecast
                        forecast = model.predict(future)
                        
                        # Show last few forecast results
                        st.subheader("Forecast Results")
                        st.dataframe(forecast[['ds', 'yhat']].tail(), use_container_width=True)

                        # Plot forecast
                        fig_forecast = plot_plotly(model, forecast)
                        fig_forecast.update_layout(title="Forecast vs Historical Data")
                        st.plotly_chart(fig_forecast, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error in forecasting: {str(e)}")

if __name__ == "__main__":
    main()
