import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
from datetime import datetime
import warnings
import plotly.express as px

warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(page_title="Time Series Forecast Pro", layout="wide")

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
    st.title("📈 Time Series Forecasting Application")
    st.markdown("""<p style="font-size: 18px; color: #555555;">Upload your time series data or enter manually to generate forecasts.</p>""", unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Data input options
    with st.expander("📤 Data Input Options", expanded=True):
        input_method = st.radio("Choose data input method:", 
                              ("Upload CSV", "Manual Input"), horizontal=True)

        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your time series CSV file", 
                                           type=["csv"], help="Select your CSV file here.")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("Data uploaded successfully!")

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
        st.dataframe(df.head(), use_container_width=True)

        # Column selection
        cols = df.columns.tolist()
        date_col, value_col = st.columns(2)
        with date_col:
            date_column = st.selectbox("Select Date Column", options=cols)
        with value_col:
            value_column = st.selectbox("Select Value Column", options=cols)

        # Convert date column
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            st.error(f"Error converting date column to datetime: {str(e)}")
            return

        # Convert value column to numeric if necessary
        try:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        except Exception as e:
            st.error(f"Error converting value column to numeric: {str(e)}")
            return

        # Check if the column is numeric
        if not np.issubdtype(df[value_column].dtype, np.number):
            st.error(f"Selected value column '{value_column}' is not numeric.")
            return

        # Time series visualization
        st.subheader("Time Series Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_column], y=df[value_column], 
                                 mode='lines+markers', name='Actual', line=dict(color='blue')))
        fig.update_layout(
            title="Time Series Overview",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Additional Visualizations
        st.subheader("Additional Insights")
        
        # Histogram of Values
        st.markdown("### Distribution of Values")
        fig_hist = px.histogram(df, x=value_column, nbins=30, title="Value Distribution")
        fig_hist.update_traces(marker_color="purple", opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box Plot for Outlier Detection
        st.markdown("### Box Plot for Outlier Detection")
        fig_box = px.box(df, y=value_column, title="Box Plot of Values")
        st.plotly_chart(fig_box, use_container_width=True)

        # Rolling Mean and Standard Deviation
        st.markdown("### Rolling Statistics")
        df['rolling_mean'] = df[value_column].rolling(window=30).mean()
        df['rolling_std'] = df[value_column].rolling(window=30).std()
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=df[date_column], y=df['rolling_mean'], name='Rolling Mean', line=dict(color='green')))
        fig_roll.add_trace(go.Scatter(x=df[date_column], y=df['rolling_std'], name='Rolling Std', line=dict(color='red')))
        fig_roll.update_layout(
            title="Rolling Mean and Standard Deviation",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_roll, use_container_width=True)

        # Data decomposition
        with st.expander("Advanced Analysis"):
            st.markdown("### Time Series Decomposition")
            period = st.slider("Seasonality Period", 1, 365, 30, help="Adjust this to match your data's seasonality.")
            
            try:
                decomposition_fig = go.Figure()
                decomposition_fig.add_trace(go.Scatter(x=df[date_column], y=df[value_column], name='Original'))
                decomposition_fig.add_trace(go.Scatter(x=df[date_column], y=df[value_column].rolling(period).mean(), name='Trend'))
                decomposition_fig.add_trace(go.Scatter(x=df[date_column], y=df[value_column].diff().rolling(period).mean(), name='Seasonality'))
                decomposition_fig.update_layout(
                    title="Trend and Seasonality Analysis",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_white",
                    hovermode="x unified"
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
                                  index=0, help="Choose the frequency of your forecast.")
            with col3:
                confidence = st.slider("Confidence Interval", 70, 99, 95, help="Set the confidence level for your forecast.")
            
            if st.form_submit_button("Generate Forecast"):
                with st.spinner("Training model and generating forecast..."):
                    try:
                        # Prepare data for Prophet
                        prophet_df = df[[date_column, value_column]]
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
                        future = model.make_future_dataframe(periods=periods, freq=freq)
                        
                        # Generate forecast
                        forecast = model.predict(future)
                        
                        # Show forecast results
                        st.subheader("Forecast Results")
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), use_container_width=True)

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
                            mae = np.mean(np.abs(forecast.yhat - prophet_df.y))
                            st.metric("MAE", f"{mae:.2f}")
                        with col2:
                            rmse = np.sqrt(np.mean((forecast.yhat - prophet_df.y)**2))
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col3:
                            st.metric("Forecast Horizon", f"{periods} {freq}")

                    except Exception as e:
                        st.error(f"Error in forecasting: {str(e)}")

if __name__ == "__main__":
    main()
