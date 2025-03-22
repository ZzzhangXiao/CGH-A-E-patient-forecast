import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from nixtla import NixtlaClient

st.set_page_config(layout="wide")
cgh_logo = Image.open('collage.png')
st.image(cgh_logo, width=300)
st.title("A&E Patient Arrival Forecast")

def run_forecast_model(model_name, train_data, test_data, forecast_days, model_params=None):
    y_true = test_data["y"].reset_index(drop=True) if "y" in test_data else None
    forecast = None
    metrics = {}

    if model_name == "Seasonal ARIMA (Statistical)":
        p_, d_, q_ = model_params.get("order", (0, 0, 0))
        P_, D_, Q_, m = model_params.get("seasonal_order", (1, 0, 1, 7))
        model = SARIMAX(train_data["y"],
                        order=(p_, d_, q_),
                        seasonal_order=(P_, D_, Q_, m),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        forecast = fitted_model.forecast(steps=forecast_days)

    elif model_name == "TimeGPT":
        nixtla_client = NixtlaClient(api_key=model_params.get("api_key"))
        full_forecast = nixtla_client.forecast(
            df=train_data,
            model="timegpt-1",
            h=forecast_days,
            time_col="ds",
            target_col="y"
        )
        forecast = full_forecast["TimeGPT"]

    elif model_name == "Prophet":
        model = Prophet()
        model.fit(train_data[["ds", "y"]])
        future = pd.DataFrame({"ds": test_data["ds"]})
        forecast_df = model.predict(future)
        forecast = forecast_df["yhat"]

    elif model_name == "XGBoost":
        lags = forecast_days

        def create_lag_features(df, lags):
            df = df.copy()
            for lag in range(1, lags + 1):
                df[f"lag_{lag}"] = df["y"].shift(lag)
            return df.dropna()

        train_data_lagged = create_lag_features(train_data, lags)
        X_train = train_data_lagged.drop(columns=["ds", "y"])
        y_train = train_data_lagged["y"]

        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)

        recent_history = train_data["y"].tolist()[-lags:]
        forecast = []

        for step in range(forecast_days):
            X_input = np.array(recent_history[-lags:]).reshape(1, -1)
            y_pred = model.predict(X_input)[0]
            forecast.append(y_pred)
            recent_history.append(y_pred)

        forecast = pd.Series(forecast)

    if forecast is not None and y_true is not None:
        forecast = pd.Series(forecast).reset_index(drop=True)
        rmse = np.sqrt(mean_squared_error(y_true, forecast))
        mae = mean_absolute_error(y_true, forecast)
        mape = mean_absolute_percentage_error(y_true, forecast) * 100

        actual_direction = np.sign(y_true.diff().fillna(0))
        predicted_direction = np.sign(forecast.diff().fillna(0))
        directional_accuracy = (actual_direction == predicted_direction).mean() * 100

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape,
            "Directional Accuracy (%)": directional_accuracy
        }

    return pd.Series(forecast).reset_index(drop=True), metrics

# UI
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("1. Data Upload")
    st.write("Expected Data Format")
    st.table(pd.DataFrame({
        'A&E Admit Date': ['2025-02-01', '2025-02-02', '2025-02-03'],
        'A&E Case Number': [400, 500, 600]
    }))
    uploaded_file = st.file_uploader("Upload your time series data", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        if 'A&E Admit Date' in df.columns:
            df['A&E Admit Date'] = pd.to_datetime(df['A&E Admit Date']).dt.strftime('%Y-%m-%d')
        st.dataframe(df.head())

with col2:
    if uploaded_file is not None:
        st.subheader("2. Forecast Parameters")
        param1, param2 = st.columns(2)
        with param1:
            train_days = st.number_input("Training days", min_value=1, value=30)
        with param2:
            forecast_days = st.number_input("Forecast horizon", min_value=1, value=7)
        model_choices = st.multiselect("Select Forecasting Models", ["TimeGPT", "Seasonal ARIMA (Statistical)", "Prophet", "XGBoost"], default=["TimeGPT"])

        # Show API key input if TimeGPT is selected
        timegpt_api_key = None
        if "TimeGPT" in model_choices:
            timegpt_api_key = st.text_input("Enter your TimeGPT API Key", type="password")

        if st.button("Compare Models"):
            df['A&E Admit Date'] = pd.to_datetime(df['A&E Admit Date'])
            df = df.sort_values('A&E Admit Date')
            df_daily = df.rename(columns={'A&E Admit Date': 'ds', 'A&E Case Number': 'y'})
            train_data = df_daily.iloc[-(train_days + forecast_days):-forecast_days].copy()
            test_data = df_daily.iloc[-forecast_days:].copy()
            results = {}
            for model_choice in model_choices:
                model_params = {}
                if model_choice == "Seasonal ARIMA (Statistical)":
                    model_params = {"order": (0, 0, 0), "seasonal_order": (1, 0, 1, 7)}
                elif model_choice == "TimeGPT":
                    if not timegpt_api_key:
                        results[model_choice] = {"error": "Please enter a valid TimeGPT API key."}
                        continue
                    model_params = {"api_key": timegpt_api_key}
                elif model_choice == "XGBoost":
                    model_params = {
                        "n_estimators": 50, "max_depth": 3, "learning_rate": 0.01,
                        "subsample": 0.8, "colsample_bytree": 1.0, "random_state": 42
                    }
                try:
                    forecast, metrics = run_forecast_model(
                        model_name=model_choice,
                        train_data=train_data.copy(),
                        test_data=test_data.copy(),
                        forecast_days=forecast_days,
                        model_params=model_params
                    )
                    results[model_choice] = {"forecast": forecast, "metrics": metrics}
                except Exception as e:
                    results[model_choice] = {"error": str(e)}

            if results:
                st.subheader("3. Model Comparison")
                tabs = st.tabs(results.keys())
                metric_table = []
                for i, model_name in enumerate(results):
                    with tabs[i]:
                        if "error" in results[model_name]:
                            st.error(f"{model_name} Error: {results[model_name]['error']}")
                        else:
                            forecast = results[model_name]["forecast"]
                            metrics = results[model_name]["metrics"]
                            metric_table.append({"Model": model_name, **metrics})

                            st.markdown(f"### {model_name} Forecast")
                            recent_train = df_daily.iloc[-train_days:].copy()
                            recent_train_dates = recent_train['ds']
                            recent_train_values = recent_train['y']
                            forecast_dates = test_data['ds'].reset_index(drop=True)

                            final_train = df_daily.iloc[-train_days:].copy()
                            future_start = df_daily['ds'].max() + pd.Timedelta(days=1)
                            future_dates = pd.date_range(start=future_start, periods=forecast_days)
                            try:
                                future_forecast, _ = run_forecast_model(
                                    model_name=model_name,
                                    train_data=final_train,
                                    test_data=pd.DataFrame({"ds": future_dates}),
                                    forecast_days=forecast_days,
                                    model_params=model_params
                                )
                            except:
                                future_forecast = pd.Series([np.nan] * forecast_days)

                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(recent_train_dates, recent_train_values, label="Recent History", color='gray', alpha=0.5)
                            ax.plot(forecast_dates, test_data['y'].reset_index(drop=True), label="Actual", marker='o', color='black')
                            ax.plot(forecast_dates, forecast, label=f"{model_name} (Test Set)", marker='x', linestyle='--', color='blue')
                            ax.plot(future_dates, future_forecast, label=f"{model_name} (Next {forecast_days} Days)", marker='s', linestyle='dotted', color='red')
                            if len(forecast) > 0 and len(future_forecast) > 0:
                                join_x = [forecast_dates.iloc[-1], future_dates[0]]
                                join_y = [forecast.iloc[-1], future_forecast.iloc[0]]
                                ax.plot(join_x, join_y, color='blue', linestyle='--')

                            plt.setp(ax.get_xticklabels(), rotation=45)
                            ax.legend(fontsize=8, loc='upper right', frameon=False)
                            ax.grid()
                            fig.tight_layout()
                            st.pyplot(fig)

                            metrics_col, summary_col = st.columns([2, 3])
                            with metrics_col:
                                st.markdown("### Evaluation Metrics")
                                for key, value in metrics.items():
                                    if "Accuracy" in key:
                                        st.markdown(f"**{key}**: <span style='color:red'>{value:.1f}</span> (the closer to 100 the better)", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"**{key}**: <span style='color:red'>{value:.1f}</span> (the lower the better)", unsafe_allow_html=True)

                            with summary_col:
                                st.markdown("### Model Metrics Summary")
                                st.dataframe(pd.DataFrame(metric_table).set_index("Model").style.format("{:.2f}"))

                            st.markdown("### Forecast Table")
                            future_only_df = pd.DataFrame({
                                "Date": future_dates.strftime('%Y-%m-%d'),
                                "Forecast": np.round(future_forecast).astype(int)
                            })
                            st.dataframe(future_only_df)
