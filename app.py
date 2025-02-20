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
    .st-bw {background-color: #000000; color: white; padding: 10px;}
    .css-18e3th9 {padding: 2rem 1rem;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 15px;}
    .stTextInput>div>div>input {background-color: #e8f0fe; border-radius: 3px;}
    @media (max-width: 600px) {
        .css-18e3th9 {padding: 1rem;}
    }
</style>
""", unsafe_allow_html=True)

def prepare_data(df, date_column, value_column):
    """Prepare data for Prophet model."""
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])  # Remove rows with invalid dates
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        return df.sort_values(by=date_column)
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None

def plot_time_series(df, date_column, value_column):
    """Plot the time series data."""
    fig = px.line(df, x=date_column, y=value_column, title="Historical Data")
    st.plotly_chart(fig, use_container_width=True)

def forecast_data(df, date_column, value_column, days_to_forecast):
    """Perform forecasting using Prophet."""
    prophet_df = df[[date_column, value_column]].rename(columns={date_column: 'ds', value_column: 'y'})
    
    model = Prophet()
    with st.spinner("Generating forecast..."):
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=days_to_forecast)
        forecast = model.predict(future)
    
    return model, forecast

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

        # Prepare data
        df = prepare_data(df, date_column, value_column)
        if df is None:
            return

        # Time series visualization
        st.subheader("Time Series Overview")
        plot_time_series(df, date_column, value_column)

        # Simple Forecasting
        st.subheader("🔮 Forecast")
        with st.form("forecast_settings"):
            days_to_forecast = st.number_input("Days to Forecast", min_value=1, max_value=365, value=30)
            if st.form_submit_button("Predict"):
                model, forecast = forecast_data(df, date_column, value_column, days_to_forecast)

                # Show last few forecast results
                st.subheader("Forecast Results")
                st.dataframe(forecast[['ds', 'yhat']].tail(), use_container_width=True)

                # Plot forecast
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(title="Forecast vs Historical Data")
                st.plotly_chart(fig_forecast, use_container_width=True)

if __name__ == "__main__":
    main()
