import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

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

def main():
    st.title("📈 Simple Time Series Forecasting")
    st.markdown("""<p style="font-size: 18px; color: #555555;">Upload your time series data or enter manually to generate forecasts.</p>""", unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Data input options
    with st.expander("📤 Data Input", expanded=True):
        input_method = st.radio("Choose data input method:", ("Upload CSV", "Manual Input"), horizontal=True)

        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])
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
                        "date": st.column_config.DateColumn("Date", format="DD-MM-YYYY"),
                        "value": st.column_config.NumberColumn("Value")
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

        # Convert to datetime, handling potential errors
        try:
            df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y', errors='coerce')
            df = df.dropna(subset=[date_column])
        except Exception as e:
            st.error(f"Error converting date column to datetime: {str(e)}")
            return

        # Convert value column to numeric
        try:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            df = df.dropna(subset=[value_column])
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
                        prophet_df = df[[date_column, value_column]].rename(columns={date_column: 'ds', value_column: 'y'})

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
