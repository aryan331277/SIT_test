
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(page_title="Time Series Forecast Pro", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2a3f5f;}
    .st-bw {background-color: #000000;}
    .css-18e3th9 {padding: 2rem 5rem;}
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    st.title("📈 Time Series Forecasting Application")
    st.markdown("Upload your time series data or enter manually to generate forecasts")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Data input options
    with st.expander("📤 Data Input Options", expanded=True):
        input_method = st.radio("Choose data input method:", 
                              ("Upload CSV", "Manual Input"))

        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your time series CSV file", 
                                           type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df

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
                    }
                )
                if st.form_submit_button("Save Manual Data"):
                    st.session_state.df = manual_data

    # Main analysis section
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Column selection
        cols = df.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Select Date Column", options=cols)
        with col2:
            value_col = st.selectbox("Select Value Column", options=cols)

        # Convert date column
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            st.error("Could not convert selected date column to datetime format")
            return

        # Time series visualization
        st.subheader("Time Series Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], 
                               mode='lines+markers', name='Actual'))
        fig.update_layout(
            title="Time Series Overview",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data decomposition
        with st.expander("Advanced Analysis"):
            st.markdown("### Time Series Decomposition")
            period = st.slider("Seasonality Period", 1, 365, 30)
            
            try:
                decomposition_fig = go.Figure()
                decomposition_fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[value_col], name='Original'))
                decomposition_fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[value_col].rolling(period).mean(), 
                    name='Trend'))
                decomposition_fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[value_col].diff().rolling(period).mean(), 
                    name='Seasonality'))
                decomposition_fig.update_layout(
                    title="Trend and Seasonality Analysis",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_white"
                )
                st.plotly_chart(decomposition_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in decomposition: {str(e)}")

        # Forecasting section
        st.subheader("🔮 Forecasting Configuration")
        with st.form("forecast_settings"):
            col1, col2, col3 = st.columns(3)
            with col1:
                periods = st.number_input("Forecast Periods", 1, 365, 30)
            with col2:
                freq = st.selectbox("Frequency", 
                                  ["D", "W", "M", "Q", "Y"], 
                                  index=0)
            with col3:
                confidence = st.slider("Confidence Interval", 70, 99, 95)
            
            if st.form_submit_button("Generate Forecast"):
                with st.spinner("Training model and generating forecast..."):
                    try:
                        # Prepare data for Prophet
                        prophet_df = df[[date_col, value_col]]
                        prophet_df.columns = ['ds', 'y']

                        # Model training
                        model = Prophet(
                            yearly_seasonality='auto',
                            weekly_seasonality='auto',
                            daily_seasonality=False,
                            interval_width=confidence/100
                        )
                        model.fit(prophet_df)

                        # Generate future dataframe
                        future = model.make_future_dataframe(
                            periods=periods, 
                            freq=freq
                        )
                        
                        # Generate forecast
                        forecast = model.predict(future)
                        
                        # Show forecast results
                        st.subheader("Forecast Results")
                        st.dataframe(forecast.tail(), use_container_width=True)

                        # Plot forecast
                        st.markdown("### Forecast Visualization")
                        fig1 = plot_plotly(model, forecast)
                        st.plotly_chart(fig1, use_container_width=True)

                        # Plot components
                        st.markdown("### Forecast Components")
                        fig2 = plot_components_plotly(model, forecast)
                        st.plotly_chart(fig2, use_container_width=True)

                        # Performance metrics
                        st.markdown("### Model Performance")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", 
                                    f"{np.mean(np.abs(forecast.yhat - prophet_df.y)):.2f}")
                        with col2:
                            st.metric("RMSE", 
                                    f"{np.sqrt(np.mean((forecast.yhat - prophet_df.y)**2)):.2f}")
                        with col3:
                            st.metric("Forecast Horizon", 
                                    f"{periods} {freq}")

                    except Exception as e:
                        st.error(f"Error in forecasting: {str(e)}")

if __name__ == "__main__":
    main()
