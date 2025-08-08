import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import io
import openpyxl
from datetime import timedelta

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Cambodia CleanTech Analytics",
    layout="wide",
    page_icon="🌿",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --primary-blue: #1e40af;
        --secondary-blue: #3b82f6;
        --accent-green: #059669;
        --warning-orange: #d97706;
        --danger-red: #dc2626;
        --neutral-gray: #6b7280;
        --light-gray: #f8fafc;
        --white: #ffffff;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .metric-container {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: var(--shadow);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--neutral-gray);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .alert-success {
        background-color: #dcfce7;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fef3c7;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .alert-info {
        background-color: #dbeafe;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-blue);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: var(--primary-blue);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: var(--secondary-blue);
        box-shadow: var(--shadow-lg);
    }
    .sidebar .stSelectbox>div>div {
        background-color: var(--white);
    }
    .professional-title {
        text-align: center;
        color: var(--primary-blue);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .professional-subtitle {
        text-align: center;
        color: var(--neutral-gray);
        font-size: 1.125rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Data Generation
@st.cache_data
def generate_cambodia_dataset(n_months=36, seed=42):
    np.random.seed(seed)
    months = np.arange(n_months)
    dates = pd.date_range(start="2021-01-01", periods=n_months, freq="M")
    
    # Energy consumption with seasonal patterns
    base_energy = 1200
    trend = base_energy * (1.04) ** (months / 12)
    seasonal = 100 * np.sin(2 * np.pi * months / 12)
    noise = np.random.normal(0, 50, n_months)
    energy_use = np.clip(trend + seasonal + noise, 800, 2000)
    
    # Production rate
    production_base = 250
    production_trend = production_base * (1.035) ** (months / 12)
    production_seasonal = 30 * np.cos(2 * np.pi * months / 12)
    production_noise = np.random.normal(0, 20, n_months)
    production_rate = np.clip(production_trend + production_seasonal + production_noise, 150, 400)
    
    # Waste generation
    waste_base = 80
    waste = np.clip(waste_base + 0.1 * production_rate + np.random.normal(0, 15, n_months), 40, 150)
    
    # CO2 emissions
    co2_base = 0.65 * energy_use + 0.12 * production_rate + 0.3 * waste
    co2_seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * months / 12 + np.pi/4)
    co2_emissions = np.clip(co2_base * co2_seasonal_factor + np.random.normal(0, 25, n_months), 200, 1500)
    
    df = pd.DataFrame({
        'date': dates,
        'energy_use': energy_use,
        'production_rate': production_rate,
        'waste': waste,
        'co2_emissions': co2_emissions
    })
    
    df['energy_intensity'] = df['co2_emissions'] / df['energy_use']
    df['production_efficiency'] = df['production_rate'] / df['energy_use']
    df['waste_intensity'] = df['waste'] / df['production_rate']
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    return df

# Analytics Class
class CleanTechAnalytics:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.models = {}

    def calculate_kpis(self):
        current_period = self.df.iloc[-12:]
        previous_period = self.df.iloc[-24:-12] if len(self.df) >= 24 else self.df.iloc[:-12]
        
        kpis = {
            'total_emissions': self.df['co2_emissions'].sum(),
            'avg_monthly_emissions': self.df['co2_emissions'].mean(),
            'emissions_trend': self.calculate_trend('co2_emissions'),
            'energy_efficiency': self.df['production_rate'].sum() / self.df['energy_use'].sum(),
            'waste_per_unit': self.df['waste'].sum() / self.df['production_rate'].sum(),
            'current_emission_rate': current_period['co2_emissions'].mean(),
            'previous_emission_rate': previous_period['co2_emissions'].mean() if len(previous_period) > 0 else 0,
        }
        
        kpis['emission_change'] = ((kpis['current_emission_rate'] - kpis['previous_emission_rate']) / 
                                 kpis['previous_emission_rate']) * 100 if kpis['previous_emission_rate'] > 0 else 0
        return kpis

    def calculate_trend(self, column, periods=12):
        recent_data = self.df[column].tail(periods)
        if len(recent_data) < 2:
            return 0
        x = np.arange(len(recent_data))
        coeffs = np.polyfit(x, recent_data, 1)
        return coeffs[0]

    def build_predictive_models(self, forecast_months=12):
        features = ['energy_use', 'production_rate', 'waste', 'month', 'energy_intensity', 'production_efficiency']
        target = 'co2_emissions'
        
        X = self.df[features].copy()
        y = self.df[target].values
        
        # Enhanced feature engineering
        for col in ['energy_use', 'production_rate', 'co2_emissions']:
            X[f'{col}_lag1'] = self.df[col].shift(1)
            X[f'{col}_lag3'] = self.df[col].shift(3)
            X[f'{col}_rolling_mean'] = self.df[col].rolling(window=3).mean()
            X[f'{col}_rolling_std'] = self.df[col].rolling(window=3).std()
        
        X = X.dropna()
        y = y[-len(X):]
        
        if len(X) < 10:
            return None, None, {}, None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
        }
        
        model_results = {}
        best_model = None
        best_score = float('-inf')
        
        for name, model in models.items():
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'predictions': y_pred,
                'actuals': y_test
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = (name, model)
        
        # Generate future predictions
        future_predictions = self.generate_future_predictions(best_model[1], X, forecast_months, features)
        
        return best_model, X.columns.tolist(), model_results, future_predictions

    def generate_future_predictions(self, model, X, forecast_months, base_features):
        last_date = self.df['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=30), periods=forecast_months, freq='M')
        
        future_X = []
        future_predictions = []
        
        last_observation = X.iloc[-1].copy()
        scaler_features = X.columns
        
        for i in range(forecast_months):
            new_row = last_observation.copy()
            new_row['month'] = future_dates[i].month
            
            # Determine lag values
            # For lag1, use the previous forecast or the last historical value
            if i > 0:
                new_row['energy_use_lag1'] = future_X[i-1]['energy_use']
                new_row['production_rate_lag1'] = future_X[i-1]['production_rate']
                new_row['co2_emissions_lag1'] = future_predictions[i-1]
            else: # Use historical data for the first iteration
                new_row['energy_use_lag1'] = X['energy_use'].iloc[-1]
                new_row['production_rate_lag1'] = X['production_rate'].iloc[-1]
                new_row['co2_emissions_lag1'] = self.df['co2_emissions'].iloc[-1]

            # For lag3, use the value from 3 periods ago
            if i >= 3:
                new_row['energy_use_lag3'] = future_X[i-3]['energy_use']
                new_row['production_rate_lag3'] = future_X[i-3]['production_rate']
                new_row['co2_emissions_lag3'] = future_predictions[i-3]
            else: # Use historical data for the first three iterations
                new_row['energy_use_lag3'] = X['energy_use_lag3'].iloc[-1] if 'energy_use_lag3' in X.columns else X['energy_use'].iloc[-3]
                new_row['production_rate_lag3'] = X['production_rate_lag3'].iloc[-1] if 'production_rate_lag3' in X.columns else X['production_rate'].iloc[-3]
                new_row['co2_emissions_lag3'] = X['co2_emissions_lag3'].iloc[-1] if 'co2_emissions_lag3' in X.columns else self.df['co2_emissions'].iloc[-3]

            # Update rolling statistics (requires at least 3 points)
            if i >= 2:
                recent_energy = [future_X[j]['energy_use'] for j in range(i-2, i)]
                recent_prod = [future_X[j]['production_rate'] for j in range(i-2, i)]
                recent_co2 = [future_predictions[j] for j in range(i-2, i)]
            elif i == 1:
                recent_energy = [X['energy_use'].iloc[-1]] + [new_row['energy_use']]
                recent_prod = [X['production_rate'].iloc[-1]] + [new_row['production_rate']]
                recent_co2 = [self.df['co2_emissions'].iloc[-1]] + [new_row['co2_emissions']]
            else: # i == 0
                recent_energy = self.df['energy_use'].iloc[-3:].tolist()
                recent_prod = self.df['production_rate'].iloc[-3:].tolist()
                recent_co2 = self.df['co2_emissions'].iloc[-3:].tolist()

            new_row['energy_use_rolling_mean'] = np.mean(recent_energy)
            new_row['energy_use_rolling_std'] = np.std(recent_energy) if len(recent_energy) > 1 else 0
            new_row['production_rate_rolling_mean'] = np.mean(recent_prod)
            new_row['production_rate_rolling_std'] = np.std(recent_prod) if len(recent_prod) > 1 else 0
            new_row['co2_emissions_rolling_mean'] = np.mean(recent_co2)
            new_row['co2_emissions_rolling_std'] = np.std(recent_co2) if len(recent_co2) > 1 else 0
            
            # Predict the next value
            future_X_df_to_transform = pd.DataFrame([new_row])[scaler_features]
            future_X_scaled = self.scaler.transform(future_X_df_to_transform) if isinstance(model, LinearRegression) else future_X_df_to_transform
            
            next_prediction = model.predict(future_X_scaled)[0]
            
            future_predictions.append(next_prediction)
            new_row['co2_emissions'] = next_prediction
            future_X.append(new_row)
            last_observation = new_row
                
        return pd.DataFrame({
            'date': future_dates,
            'co2_emissions_predicted': future_predictions
        })

    def optimize_technology_portfolio(self, budget, reduction_target):
        technologies = {
            'Solar PV Systems': {
                'max_investment': budget * 0.8,
                'co2_reduction_per_dollar': 0.002,
                'implementation_complexity': 0.3,
                'payback_period': 6
            },
            'Energy Efficiency': {
                'max_investment': budget * 0.6,
                'co2_reduction_per_dollar': 0.0035,
                'implementation_complexity': 0.2,
                'payback_period': 3
            },
            'Waste Management': {
                'max_investment': budget * 0.4,
                'co2_reduction_per_dollar': 0.0015,
                'implementation_complexity': 0.4,
                'payback_period': 4
            },
            'Electric Vehicles': {
                'max_investment': budget * 0.5,
                'co2_reduction_per_dollar': 0.0025,
                'implementation_complexity': 0.6,
                'payback_period': 8
            },
            'CCUS Technology': {
                'max_investment': budget * 0.3,
                'co2_reduction_per_dollar': 0.001,
                'implementation_complexity': 0.8,
                'payback_period': 12
            }
        }
        
        n_tech = len(technologies)
        tech_names = list(technologies.keys())
        c = [-tech['co2_reduction_per_dollar'] for tech in technologies.values()]
        A_ub = [[1] * n_tech]
        b_ub = [budget]
        bounds = [(0, tech['max_investment']) for tech in technologies.values()]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            investments = result.x
            total_reduction = -result.fun
            
            portfolio_df = pd.DataFrame({
                'Technology': tech_names,
                'Investment (USD)': investments,
                'Investment (%)': (investments / budget) * 100,
                'CO2 Reduction (tons/year)': [inv * tech['co2_reduction_per_dollar'] 
                                           for inv, tech in zip(investments, technologies.values())],
                'Payback Period (years)': [tech['payback_period'] for tech in technologies.values()],
                'Implementation Complexity': [tech['implementation_complexity'] for tech in technologies.values()]
            })
            
            portfolio_df = portfolio_df[portfolio_df['Investment (USD)'] > 100].reset_index(drop=True)
            return portfolio_df, total_reduction
        return pd.DataFrame(), 0

# Main Application
def main():
    st.markdown("""
        <div class="professional-title">🌿 Cambodia CleanTech Analytics</div>
        <div class="professional-subtitle">
            Advanced Data Analytics for Sustainable Development • Ministry of Environment Partnership
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        data_source = st.selectbox(
            "Data Source",
            ["Generated Cambodia Dataset", "Upload CSV File"],
            help="Choose your data source for analysis"
        )
        st.markdown("### 🎯 Analysis Parameters")
        budget = st.number_input(
            "Investment Budget (USD)",
            min_value=100000,
            max_value=10000000,
            value=2000000,
            step=100000,
            format="%d"
        )
        reduction_target = st.slider(
            "CO₂ Reduction Target (%)",
            min_value=5,
            max_value=50,
            value=27,
            help="Based on Cambodia's NDC commitment"
        )
        forecast_months = st.selectbox(
            "Forecast Horizon",
            [6, 12, 18, 24],
            index=1
        )
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV file with columns: energy_use, production_rate, waste, co2_emissions",
            type=['csv']
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['energy_use', 'production_rate', 'waste', 'co2_emissions']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Missing required columns: {required_cols}")
                    return
                df['date'] = pd.date_range(start="2021-01-01", periods=len(df), freq="M")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("Please upload a CSV file to proceed.")
            return
    else:
        df = generate_cambodia_dataset()
    
    analytics = CleanTechAnalytics(df)
    kpis = analytics.calculate_kpis()
    
    # KPI Dashboard
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "red" if kpis['emission_change'] > 0 else "green"
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Monthly Avg Emissions</div>
                <div class="metric-value">{kpis['current_emission_rate']:.0f}</div>
                <div class="metric-delta" style="color: {delta_color}">
                    {kpis['emission_change']:+.1f}% from previous period
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Energy Efficiency</div>
                <div class="metric-value">{kpis['energy_efficiency']:.3f}</div>
                <div class="metric-delta">units per MWh</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Waste Intensity</div>
                <div class="metric-value">{kpis['waste_per_unit']:.3f}</div>
                <div class="metric-delta">tons per unit produced</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        trend_direction = "↗️" if kpis['emissions_trend'] > 0 else "↘️"
        trend_color = "red" if kpis['emissions_trend'] > 0 else "green"
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Emissions Trend</div>
                <div class="metric-value" style="color: {trend_color}">{trend_direction}</div>
                <div class="metric-delta">{abs(kpis['emissions_trend']):.1f} tons/month</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Time Series Visualization
    st.markdown('<div class="section-header">📈 Time Series Analysis</div>', unsafe_allow_html=True)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CO₂ Emissions Over Time', 'Energy Consumption', 
                       'Production Rate', 'Waste Generation'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['co2_emissions'], name='CO₂ Emissions',
                  line=dict(color='#dc2626', width=2)), row=1, col=1
    )
    z = np.polyfit(range(len(df)), df['co2_emissions'], 1)
    trend_line = np.poly1d(z)(range(len(df)))
    fig.add_trace(
        go.Scatter(x=df['date'], y=trend_line, name='Trend',
                  line=dict(color='#dc2626', width=1, dash='dash')), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['energy_use'], name='Energy Use',
                  line=dict(color='#1e40af', width=2)), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['production_rate'], name='Production',
                  line=dict(color='#059669', width=2)), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['waste'], name='Waste',
                  line=dict(color='#d97706', width=2)), row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Historical Performance Dashboard")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="CO₂ Emissions (tons)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (MWh)", row=1, col=2)
    fig.update_yaxes(title_text="Production (units)", row=2, col=1)
    fig.update_yaxes(title_text="Waste (tons)", row=2, col=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Modeling
    st.markdown('<div class="section-header">🤖 Predictive Analytics</div>', unsafe_allow_html=True)
    with st.spinner("Building predictive models..."):
        best_model_info, features, model_results, future_predictions = analytics.build_predictive_models(forecast_months)
    
    if model_results:
        model_comparison = pd.DataFrame({
            'Model': list(model_results.keys()),
            'R² Score': [results['r2'] for results in model_results.values()],
            'MAE': [results['mae'] for results in model_results.values()],
            'RMSE': [np.sqrt(results['mse']) for results in model_results.values()],
            'CV Score (mean ± std)': [f"{results['cv_score_mean']:.3f} ± {results['cv_score_std']:.3f}" 
                                    for results in model_results.values()]
        })
        
        st.markdown("**Model Performance Comparison:**")
        st.dataframe(
            model_comparison.style.format({
                'R² Score': '{:.3f}',
                'MAE': '{:.1f}',
                'RMSE': '{:.1f}'
            }).highlight_max(subset=['R² Score'], color='lightgreen'),
            use_container_width=True
        )
        
        if best_model_info:
            best_model_name, best_model = best_model_info
            st.success(f"**Best Model:** {best_model_name} (R² = {model_results[best_model_name]['r2']:.3f})")
            
            if best_model_name in ['Random Forest', 'XGBoost'] and hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df, x='Importance', y='Feature',
                    orientation='h', title=f'Feature Importance ({best_model_name})',
                    color='Importance', color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Future Predictions Visualization
            if future_predictions is not None:
                st.markdown('<div class="section-header">🔮 Future Emissions Forecast</div>', unsafe_allow_html=True)
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=df['date'], y=df['co2_emissions'],
                    name='Historical',
                    line=dict(color='#dc2626')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=future_predictions['date'], y=future_predictions['co2_emissions_predicted'],
                    name='Forecast',
                    line=dict(color='#059669', dash='dash')
                ))
                fig_forecast.update_layout(
                    title=f'CO₂ Emissions Forecast ({forecast_months} Months)',
                    xaxis_title='Date',
                    yaxis_title='CO₂ Emissions (tons)',
                    height=400
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Insufficient data for predictive modeling. Need at least 10 data points.")
    
    # Investment Optimization
    st.markdown('<div class="section-header">💼 Investment Portfolio Optimization</div>', unsafe_allow_html=True)
    portfolio_df, total_reduction = analytics.optimize_technology_portfolio(budget, reduction_target)
    
    if not portfolio_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_portfolio = px.pie(
                portfolio_df, values='Investment (USD)', names='Technology',
                title=f'Optimal Technology Portfolio (${budget:,.0f} Budget)',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_portfolio.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_portfolio, use_container_width=True)
        
        with col2:
            st.markdown("**Portfolio Summary:**")
            st.metric("Total CO₂ Reduction", f"{total_reduction:.0f} tons/year")
            st.metric("Cost per Ton CO₂", f"${budget/total_reduction:.0f}" if total_reduction > 0 else "N/A")
            st.metric("Technologies Selected", len(portfolio_df))
            
            current_emissions = kpis['current_emission_rate'] * 12
            target_reduction = current_emissions * (reduction_target / 100)
            
            if total_reduction >= target_reduction:
                st.markdown('<div class="alert-success">✅ Target reduction achievable with current budget</div>', unsafe_allow_html=True)
            else:
                shortfall = target_reduction - total_reduction
                additional_budget = shortfall * (budget / total_reduction) if total_reduction > 0 else 0
                st.markdown(f'<div class="alert-warning">⚠️ Need additional ${additional_budget:,.0f} to meet target</div>', unsafe_allow_html=True)
        
        st.markdown("**Detailed Investment Breakdown:**")
        st.dataframe(
            portfolio_df.style.format({
                'Investment (USD)': '${:,.0f}',
                'Investment (%)': '{:.1f}%',
                'CO2 Reduction (tons/year)': '{:.0f}',
                'Payback Period (years)': '{:.0f}',
                'Implementation Complexity': '{:.1f}'
            }),
            use_container_width=True
        )
    else:
        st.error("Optimization failed. Please adjust budget or parameters.")
    
    # Impact Projection
    st.markdown('<div class="section-header">🎯 Impact Projection</div>', unsafe_allow_html=True)
    projection_months = np.arange(1, forecast_months + 1)
    baseline_emissions = [kpis['current_emission_rate']] * forecast_months
    reduction_factor = np.linspace(0, total_reduction / 12, forecast_months)
    projected_emissions = [baseline - reduction for baseline, reduction in zip(baseline_emissions, reduction_factor)]
    
    projection_df = pd.DataFrame({
        'Month': projection_months,
        'Baseline Scenario': baseline_emissions,
        'With Investment': projected_emissions,
        'Reduction': [baseline - projected for baseline, projected in zip(baseline_emissions, projected_emissions)]
    })
    
    fig_projection = go.Figure()
    fig_projection.add_trace(go.Scatter(
        x=projection_df['Month'], y=projection_df['Baseline Scenario'],
        name='Baseline (No Investment)', line=dict(color='red', dash='dash')
    ))
    fig_projection.add_trace(go.Scatter(
        x=projection_df['Month'], y=projection_df['With Investment'],
        name='With Investment Portfolio', line=dict(color='green', width=3)
    ))
    target_emissions = kpis['current_emission_rate'] * (1 - reduction_target / 100)
    fig_projection.add_hline(
        y=target_emissions, line_dash="dot", line_color="orange",
        annotation_text=f"Target: {target_emissions:.0f} tons"
    )
    
    fig_projection.update_layout(
        title=f'CO₂ Emissions Projection ({forecast_months} Month Horizon)',
        xaxis_title='Month',
        yaxis_title='CO₂ Emissions (tons)',
        height=400
    )
    st.plotly_chart(fig_projection, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_baseline = sum(baseline_emissions)
        total_projected = sum(projected_emissions)
        total_reduction_projected = total_baseline - total_projected
        st.metric(
            f"Total Reduction ({forecast_months}m)",
            f"{total_reduction_projected:.0f} tons",
            f"{(total_reduction_projected/total_baseline)*100:.1f}% vs baseline"
        )
    
    with col2:
        final_month_reduction = projected_emissions[-1]
        target_gap = max(0, final_month_reduction - target_emissions)
        st.metric(
            "Target Gap (Final Month)",
            f"{target_gap:.0f} tons",
            f"{'✅ Target Met' if target_gap <= 0 else '❌ Target Missed'}"
        )
    
    with col3:
        roi_estimate = (total_reduction_projected * 50) / budget
        st.metric(
            "Estimated ROI",
            f"{roi_estimate:.1%}",
            f"@$50/ton CO₂"
        )
    
    # Risk Analysis
    st.markdown('<div class="section-header">⚠️ Risk Analysis & Recommendations</div>', unsafe_allow_html=True)
    risk_factors = []
    
    if kpis['emissions_trend'] > 0:
        risk_factors.append({
            'Risk': 'Rising Emissions Trend',
            'Level': 'High',
            'Impact': f"Emissions increasing by {kpis['emissions_trend']:.1f} tons/month",
            'Mitigation': 'Accelerate energy efficiency investments'
        })
    
    if total_reduction < (kpis['current_emission_rate'] * 12 * reduction_target / 100):
        risk_factors.append({
            'Risk': 'Insufficient Budget',
            'Level': 'Medium',
            'Impact': 'May not achieve reduction targets',
            'Mitigation': 'Seek additional funding or adjust targets'
        })
    
    if len(df) < 24:
        risk_factors.append({
            'Risk': 'Limited Historical Data',
            'Level': 'Medium',
            'Impact': 'Predictions may be less reliable',
            'Mitigation': 'Improve data collection and monitoring'
        })
    
    avg_complexity = portfolio_df['Implementation Complexity'].mean() if not portfolio_df.empty else 0
    if avg_complexity > 0.6:
        risk_factors.append({
            'Risk': 'High Implementation Complexity',
            'Level': 'High',
            'Impact': 'Projects may face delays or cost overruns',
            'Mitigation': 'Phase implementation and build local capacity'
        })
    
    if risk_factors:
        risk_df = pd.DataFrame(risk_factors)
        def color_risk_level(val):
            if val == 'High':
                return 'background-color: #fee2e2; color: #991b1b'
            elif val == 'Medium':
                return 'background-color: #fef3c7; color: #92400e'
            else:
                return 'background-color: #dcfce7; color: #166534'
        st.dataframe(
            risk_df.style.applymap(color_risk_level, subset=['Level']),
            use_container_width=True
        )
    else:
        st.markdown('<div class="alert-success">✅ No significant risks identified</div>', unsafe_allow_html=True)
    
    # Action Plan
    st.markdown('<div class="section-header">📋 Recommended Action Plan</div>', unsafe_allow_html=True)
    action_items = []
    
    if not portfolio_df.empty:
        priority_techs = portfolio_df[
            (portfolio_df['Implementation Complexity'] < 0.5) & 
            (portfolio_df['Investment (%)'] > 15)
        ]['Technology'].tolist()
        
        if priority_techs:
            action_items.append({
                'Priority': '1 - Immediate',
                'Action': f"Implement {', '.join(priority_techs)}",
                'Timeline': '0-6 months',
                'Expected Impact': 'Quick wins, build momentum'
            })
    
    action_items.append({
        'Priority': '2 - Short-term',
        'Action': 'Establish comprehensive monitoring system',
        'Timeline': '3-9 months',
        'Expected Impact': 'Improve data quality and decision-making'
    })
    
    action_items.append({
        'Priority': '3 - Medium-term',
        'Action': 'Develop local technical capacity and training programs',
        'Timeline': '6-18 months',
        'Expected Impact': 'Sustainable implementation and maintenance'
    })
    
    if total_reduction >= (kpis['current_emission_rate'] * 12 * reduction_target / 100):
        action_items.append({
            'Priority': '4 - Long-term',
            'Action': 'Expand successful technologies to other sectors',
            'Timeline': '12-36 months',
            'Expected Impact': 'Amplify impact across economy'
        })
    
    action_df = pd.DataFrame(action_items)
    st.dataframe(action_df, use_container_width=True)
    
    # Data Export
    st.markdown('<div class="section-header">📁 Data Export & Documentation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Analysis Report (Excel)"):
            report_data = {
                'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'Data Period': f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
                'Budget Allocated': f"${budget:,.0f}",
                'Reduction Target': f"{reduction_target}%",
                'Current Emissions Rate': f"{kpis['current_emission_rate']:.0f} tons/month",
                'Projected Annual Reduction': f"{total_reduction:.0f} tons",
                'Target Achievement': 'Yes' if total_reduction >= (kpis['current_emission_rate'] * 12 * reduction_target / 100) else 'No',
                'ROI Estimate': f"{roi_estimate:.1%}",
                'Key Risks': len(risk_factors),
                'Recommended Technologies': ', '.join(portfolio_df['Technology'].tolist()) if not portfolio_df.empty else 'None'
            }
            
            report_df = pd.DataFrame([report_data])
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                report_df.to_excel(writer, sheet_name='Summary', index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="Download Report Summary (Excel)",
                data=excel_data,
                file_name=f"cambodia_cleantech_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📈 Export Detailed Data (Excel)"):
            detailed_data = df.copy()
            if not portfolio_df.empty:
                detailed_data = detailed_data.join(
                    portfolio_df.set_index('Technology'), 
                    rsuffix='_portfolio'
                )
            if future_predictions is not None:
                detailed_data = pd.concat([detailed_data, future_predictions], ignore_index=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                detailed_data.to_excel(writer, sheet_name='Detailed Data', index=False)
                if not portfolio_df.empty:
                    portfolio_df.to_excel(writer, sheet_name='Portfolio', index=False)
                if future_predictions is not None:
                    future_predictions.to_excel(writer, sheet_name='Forecast', index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="Download Detailed Dataset (Excel)",
                data=excel_data,
                file_name=f"cambodia_cleantech_data_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 1rem;">
            <strong>Cambodia CleanTech Analytics Platform</strong><br>
            Advanced Data Analytics for Sustainable Development • Powered by Machine Learning & Optimization<br>
            <em>Data sources: Ministry of Environment, World Bank, International Energy Agency</em>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()