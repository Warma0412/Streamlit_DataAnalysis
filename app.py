import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from scipy import stats
from datetime import datetime, timedelta
import warnings
import io

warnings.filterwarnings('ignore')

# ============== 页面配置 ==============
st.set_page_config(
    page_title="数据分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CSS样式 ==============
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f0f9ff; border-right: 1px solid #e0f2fe; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #0369a1; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #bae6fd; }
    .metric-card { background: #f0f9ff; border: 1px solid #e0f2fe; border-radius: 8px; padding: 1rem; }
    .metric-value { font-size: 1.5rem; font-weight: 600; color: #0c4a6e; }
    .metric-label { font-size: 0.8rem; color: #64748b; }
    .value-up { color: #dc2626; font-weight: 500; }
    .value-down { color: #16a34a; font-weight: 500; }
    .stButton>button { background-color: #bae6fd; color: #0369a1; border: 1px solid #7dd3fc; border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500; }
    .stButton>button:hover { background-color: #7dd3fc; }
    .stButton>button[kind="primary"] { background-color: #fbcfe8; color: #be185d; border: 1px solid #f9a8d4; }
    .stButton>button[kind="primary"]:hover { background-color: #f9a8d4; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f8fafc; padding: 4px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; padding: 8px 16px; color: #64748b; }
    .stTabs [aria-selected="true"] { background-color: #bae6fd !important; color: #0369a1 !important; }
    .dataframe { border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }
    th { background-color: #f0f9ff !important; color: #0369a1 !important; font-weight: 600 !important; border-bottom: 2px solid #bae6fd !important; padding: 10px !important; }
    td { border-bottom: 1px solid #f1f5f9 !important; padding: 8px !important; }
    hr { border: none; height: 1px; background: #e2e8f0; margin: 1.5rem 0; }
    .algorithm-card { background: #fdf2f8; border: 1px solid #fbcfe8; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ============== 缓存函数 ==============
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        else:
            return pd.read_excel(io.BytesIO(file_bytes))
    except:
        return None

def init_session_state():
    defaults = {
        'df': None, 'df_original': None, 'file_name': None,
        'date_columns': [], 'numeric_columns': [], 'categorical_columns': [],
        'data_loaded': False, 'current_module': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def smart_format(val):
    if pd.isna(val) or val is None:
        return "-"
    try:
        num = float(val)
        if abs(num) >= 100:
            return f"{num:,.0f}"
        else:
            return f"{num:,.2f}"
    except:
        return str(val)

def detect_column_types(df):
    date_cols, numeric_cols, cat_cols = [], [], []
    for col in df.columns:
        date_keywords = ['date', 'time', '日期', '时间', 'dt', 'day', 'month', 'year']
        col_lower = col.lower()
        has_date_keyword = any(kw in col_lower for kw in date_keywords)
        try:
            threshold = 0.5 if has_date_keyword else 0.9
            converted = pd.to_datetime(df[col].astype(str), errors='coerce')
            if converted.notna().sum() / len(df) > threshold and converted.nunique() > 1:
                date_cols.append(col)
                df[col] = converted
                continue
        except:
            pass
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            try:
                converted = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')
                if converted.notna().sum() / len(df) > 0.8:
                    df[col] = converted
                    numeric_cols.append(col)
                else:
                    cat_cols.append(col)
            except:
                cat_cols.append(col)
    return df, date_cols, numeric_cols, cat_cols

# ============== 时间选择器（时间段） ==============
def render_time_range_selector(df, date_col):
    if date_col not in df.columns:
        return None, None, None, None
    dates = pd.to_datetime(df[date_col].dropna()).sort_values().unique()
    if len(dates) < 2:
        st.warning("日期数据不足")
        return None, None, None, None
    
    date_df = pd.DataFrame({'date': pd.to_datetime(dates)})
    date_df['year'] = date_df['date'].dt.year
    date_df['month'] = date_df['date'].dt.month
    years = sorted(date_df['year'].unique(), reverse=True)
    
    st.markdown("**基期（对比期）**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        base_year = st.selectbox("年", years, key="base_year")
    with col2:
        months = sorted(date_df[date_df['year'] == base_year]['month'].unique())
        base_month = st.selectbox("月", months, key="base_month")
    with col3:
        month_dates = date_df[(date_df['year'] == base_year) & (date_df['month'] == base_month)]['date'].tolist()
        base_start = st.selectbox("开始日", [d.strftime('%Y-%m-%d') for d in month_dates], index=0, key="base_start")
    with col4:
        base_end = st.selectbox("结束日", [d.strftime('%Y-%m-%d') for d in month_dates], index=len(month_dates)-1, key="base_end")
    
    st.markdown("**目标期（分析期）**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_year = st.selectbox("年", years, key="target_year")
    with col2:
        months_target = sorted(date_df[date_df['year'] == target_year]['month'].unique())
        target_month = st.selectbox("月", months_target, key="target_month")
    with col3:
        month_dates_target = date_df[(date_df['year'] == target_year) & (date_df['month'] == target_month)]['date'].tolist()
        target_start = st.selectbox("开始日", [d.strftime('%Y-%m-%d') for d in month_dates_target], index=0, key="target_start")
    with col4:
        target_end = st.selectbox("结束日", [d.strftime('%Y-%m-%d') for d in month_dates_target], index=len(month_dates_target)-1, key="target_end")
    
    st.markdown("**快捷选择**")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("最近两期", use_container_width=True):
            return dates[-2].strftime('%Y-%m-%d'), dates[-2].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')
    with col_q2:
        if st.button("最近7天 vs 前7天", use_container_width=True):
            if len(dates) >= 14:
                return dates[-14].strftime('%Y-%m-%d'), dates[-8].strftime('%Y-%m-%d'), dates[-7].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')
            else:
                st.warning("数据不足14天")
    
    return base_start, base_end, target_start, target_end

# ============== 归因引擎 ==============
class AttributionEngine:
    @staticmethod
    def calculate_contribution(df, dimension, metric, date_col, base_start, base_end, target_start, target_end):
        """计算归因数据"""
        try:
            base_start_dt = pd.to_datetime(base_start)
            base_end_dt = pd.to_datetime(base_end)
            target_start_dt = pd.to_datetime(target_start)
            target_end_dt = pd.to_datetime(target_end)
            df[date_col] = pd.to_datetime(df[date_col])
            
            base_df = df[(df[date_col] >= base_start_dt) & (df[date_col] <= base_end_dt)]
            target_df = df[(df[date_col] >= target_start_dt) & (df[date_col] <= target_end_dt)]
            
            base_data = base_df.groupby(dimension)[metric].sum()
            target_data = target_df.groupby(dimension)[metric].sum()
            
            all_dims = sorted(list(set(base_data.index) | set(target_data.index)))
            
            total_base = base_data.sum()
            total_target = target_data.sum()
            total_change = total_target - total_base
            total_change_rate = (total_change / total_base * 100) if total_base != 0 else 0
            
            results = []
            for dim in all_dims:
                base_val = base_data.get(dim, 0)
                target_val = target_data.get(dim, 0)
                change = target_val - base_val
                
                # 变动率（该维度的变动率）
                if base_val != 0:
                    change_rate = (change / base_val) * 100
                else:
                    change_rate = 0
                
                # 贡献百分比（该维度变动占总变动的比重）
                if total_change != 0:
                    contribution_pct = (change / total_change) * 100
                else:
                    contribution_pct = 0
                
                # 贡献pp = 贡献百分比 × 总计变动率 / 100
                contribution_pp = contribution_pct * total_change_rate / 100
                
                results.append({
                    '维度': dim,
                    '基期值': base_val,
                    '目标期值': target_val,
                    '变动': change,
                    '变动率': change_rate,
                    '贡献百分比': contribution_pct,
                    '贡献pp': contribution_pp
                })
            
            result_df = pd.DataFrame(results).sort_values('变动', key=abs, ascending=False)
            
            # 添加总计行
            total_row = pd.DataFrame([{
                '维度': '总计',
                '基期值': total_base,
                '目标期值': total_target,
                '变动': total_change,
                '变动率': total_change_rate,
                '贡献百分比': 100.0,
                '贡献pp': total_change_rate
            }])
            result_df = pd.concat([total_row, result_df], ignore_index=True)
            
            return result_df, total_change, total_base, total_target, total_change_rate
        except Exception as e:
            st.error(f"归因计算错误: {str(e)}")
            return None, 0, 0, 0, 0

    @staticmethod
    def cross_analysis(df, dims, metric, date_col, base_start, base_end, target_start, target_end):
        """交叉分析"""
        try:
            base_start_dt = pd.to_datetime(base_start)
            base_end_dt = pd.to_datetime(base_end)
            target_start_dt = pd.to_datetime(target_start)
            target_end_dt = pd.to_datetime(target_end)
            df[date_col] = pd.to_datetime(df[date_col])
            
            base_df = df[(df[date_col] >= base_start_dt) & (df[date_col] <= base_end_dt)]
            target_df = df[(df[date_col] >= target_start_dt) & (df[date_col] <= target_end_dt)]
            
            pivot_base = base_df.groupby(dims)[metric].sum()
            pivot_target = target_df.groupby(dims)[metric].sum()
            
            combined = pd.concat([pivot_base, pivot_target], axis=1).fillna(0)
            combined.columns = ['基期值', '目标期值']
            combined['变动'] = combined['目标期值'] - combined['基期值']
            
            total_base = combined['基期值'].sum()
            total_target = combined['目标期值'].sum()
            total_change = combined['变动'].sum()
            total_change_rate = (total_change / total_base * 100) if total_base != 0 else 0
            
            # 变动率
            combined['变动率'] = np.where(combined['基期值'] != 0, (combined['变动'] / combined['基期值']) * 100, 0)
            
            # 贡献百分比
            combined['贡献百分比'] = np.where(total_change != 0, (combined['变动'] / total_change) * 100, 0)
            
            # 贡献pp
            combined['贡献pp'] = combined['贡献百分比'] * total_change_rate / 100
            
            combined = combined.sort_values('变动', key=abs, ascending=False)
            
            # 添加总计行
            total_row = pd.DataFrame([{
                '基期值': total_base,
                '目标期值': total_target,
                '变动': total_change,
                '变动率': total_change_rate,
                '贡献百分比': 100.0,
                '贡献pp': total_change_rate
            }], index=['总计'])
            
            combined = pd.concat([total_row, combined])
            
            return combined.reset_index(), combined
        except Exception as e:
            st.error(f"交叉分析失败: {str(e)}")
            return None, None

    @staticmethod
    def trend_analysis(df, dimension, metric, date_col):
        """趋势分析"""
        try:
            trend_df = df.groupby([date_col, dimension])[metric].sum().reset_index()
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            
            for i, dim_val in enumerate(trend_df[dimension].unique()):
                data = trend_df[trend_df[dimension] == dim_val]
                fig.add_trace(go.Scatter(
                    x=data[date_col], y=data[metric],
                    mode='lines+markers', name=str(dim_val),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5)
                ))
            
            fig.update_layout(
                title=f"{dimension} 维度 {metric} 趋势",
                height=450, plot_bgcolor='white', paper_bgcolor='white',
                hovermode='x unified',
                yaxis=dict(gridcolor='#f1f5f9', tickformat=','),
                xaxis=dict(gridcolor='#f1f5f9')
            )
            
            # 增长率统计
            growth_data = []
            for dim_val in trend_df[dimension].unique():
                values = trend_df[trend_df[dimension] == dim_val].sort_values(date_col)
                if len(values) >= 2:
                    first_val = values[metric].iloc[0]
                    last_val = values[metric].iloc[-1]
                    growth_rate = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                    growth_data.append({
                        '维度': dim_val,
                        '期初值': first_val,
                        '期末值': last_val,
                        '变动': last_val - first_val,
                        '变动率': growth_rate,
                        '平均值': values[metric].mean(),
                        '最大值': values[metric].max(),
                        '最小值': values[metric].min()
                    })
            
            growth_df = pd.DataFrame(growth_data)
            
            # 添加总计行
            total_first = trend_df.groupby(date_col)[metric].sum().iloc[0]
            total_last = trend_df.groupby(date_col)[metric].sum().iloc[-1]
            total_change = total_last - total_first
            total_rate = (total_change / total_first * 100) if total_first != 0 else 0
            
            total_row = pd.DataFrame([{
                '维度': '总计',
                '期初值': total_first,
                '期末值': total_last,
                '变动': total_change,
                '变动率': total_rate,
                '平均值': trend_df.groupby(date_col)[metric].sum().mean(),
                '最大值': trend_df.groupby(date_col)[metric].sum().max(),
                '最小值': trend_df.groupby(date_col)[metric].sum().min()
            }])
            growth_df = pd.concat([total_row, growth_df], ignore_index=True)
            
            return fig, growth_df
        except Exception as e:
            st.error(f"趋势分析失败: {str(e)}")
            return None, None


# ============== 机器学习模块 ==============
ML_ALGORITHMS = {
    "聚类分析": {"name": "KMeans聚类", "desc": "将数据分为K个组，每组内的数据相似度高。用于客户分群、市场细分等场景。"},
    "异常检测": {"name": "Isolation Forest", "desc": "孤立森林算法，通过随机选择特征和分割值来识别异常点。用于欺诈检测、异常交易识别等。"},
    "特征重要性": {"name": "随机森林", "desc": "使用多棵决策树来评估各特征对预测目标的重要性。用于特征筛选、理解影响因素等。"},
    "主成分分析": {"name": "PCA", "desc": "通过线性变换将高维数据降维，保留主要信息。用于数据压缩、去除噪声、可视化等。"}
}

class MLModule:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()

    def clustering(self, features, n_clusters=3):
        try:
            X = self.df[features].dropna()
            if len(X) < n_clusters:
                return None, None, 0
            X_scaled = self.scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            X_display = X.copy()
            X_display['聚类'] = labels
            if len(features) >= 2:
                fig = px.scatter(X_display, x=features[0], y=features[1], color='聚类', title="KMeans聚类分析")
            else:
                fig = px.histogram(X_display, x=features[0], color='聚类')
            fig.update_layout(height=450)
            stats = X_display.groupby('聚类')[features].mean().round(2)
            score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            return fig, stats, score
        except Exception as e:
            st.error(f"聚类失败: {str(e)}")
            return None, None, 0

    def anomaly_detection(self, features, contamination=0.1):
        try:
            X = self.df[features].dropna()
            if len(X) < 10:
                return None, 0
            X_scaled = self.scaler.fit_transform(X)
            iso = IsolationForest(contamination=contamination, random_state=42)
            y_pred = iso.fit_predict(X_scaled)
            X_display = X.copy()
            X_display['类型'] = ['异常' if x == -1 else '正常' for x in y_pred]
            fig = px.scatter(X_display, x=features[0], y=features[1] if len(features) > 1 else features[0],
                           color='类型', color_discrete_map={'正常': '#0369a1', '异常': '#dc2626'})
            fig.update_layout(height=450)
            return fig, (y_pred == -1).sum()
        except Exception as e:
            st.error(f"异常检测失败: {str(e)}")
            return None, 0

    def feature_importance(self, target, features):
        try:
            df_clean = self.df[features + [target]].dropna()
            if len(df_clean) < 10:
                return None, 0, 0, None
            X = df_clean[features]
            y = df_clean[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            importance = pd.DataFrame({'特征': features, '重要性': model.feature_importances_}).sort_values('重要性', ascending=True)
            fig = px.bar(importance, x='重要性', y='特征', orientation='h', title="特征重要性分析")
            fig.update_layout(height=400)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return fig, r2, rmse, importance
        except Exception as e:
            st.error(f"特征重要性分析失败: {str(e)}")
            return None, 0, 0, None

    def pca_analysis(self, features, n_components=2):
        try:
            X = self.df[features].dropna()
            if len(X) < 10:
                return None, None, 0
            X_scaled = self.scaler.fit_transform(X)
            pca = PCA(n_components=min(n_components, len(features)))
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
            if X_pca.shape[1] >= 2:
                fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA分析")
            else:
                fig = px.histogram(pca_df, x='PC1', title="PCA分析")
            fig.update_layout(height=450)
            variance = pd.DataFrame({
                '主成分': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                '解释方差比例': pca.explained_variance_ratio_ * 100,
                '累积解释方差': np.cumsum(pca.explained_variance_ratio_) * 100
            })
            return fig, variance, pca.explained_variance_ratio_.sum() * 100
        except Exception as e:
            st.error(f"PCA分析失败: {str(e)}")
            return None, None, 0

# ============== 统计分析模块 ==============
class StatsModule:
    def __init__(self, df):
        self.df = df
    
    def descriptive_stats(self, columns):
        stats_list = []
        for col in columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                stats_list.append({
                    '字段': col, '样本数': len(data), '均值': data.mean(),
                    '中位数': data.median(), '标准差': data.std(),
                    '最小值': data.min(), '最大值': data.max(),
                    '25%分位数': data.quantile(0.25), '75%分位数': data.quantile(0.75),
                    '偏度': data.skew(), '峰度': data.kurtosis()
                })
        return pd.DataFrame(stats_list)
    
    def correlation(self, columns):
        corr = self.df[columns].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            text=np.round(corr.values, 2), texttemplate='%{text}',
            colorscale=[[0, '#16a34a'], [0.5, '#ffffff'], [1, '#dc2626']],
            zmid=0, zmin=-1, zmax=1
        ))
        fig.update_layout(title="相关性热力图", height=500)
        return corr, fig

# ============== 预测模块 ==============
class ForecastModule:
    def __init__(self, df):
        self.df = df
    
    def time_series_forecast(self, date_col, metric, periods=30):
        try:
            ts_data = self.df.groupby(date_col)[metric].sum().reset_index().sort_values(date_col)
            X = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data[metric].values
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            future_X = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
            predictions = model.predict(future_X)
            last_date = ts_data[date_col].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_data[date_col], y=ts_data[metric], mode='lines+markers', name='历史数据', line=dict(color='#0369a1')))
            fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='预测数据', line=dict(color='#db2777', dash='dash')))
            fig.update_layout(title=f"{metric} 趋势预测", height=450, plot_bgcolor='white', yaxis=dict(gridcolor='#f1f5f9', tickformat=','), xaxis=dict(gridcolor='#f1f5f9'))
            forecast_df = pd.DataFrame({'日期': future_dates, '预测值': predictions})
            return fig, forecast_df, model.coef_[0]
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
            return None, None, 0

# ============== 渲染函数 ==============
def render_welcome():
    st.markdown("### 数据分析平台")
    st.caption("上传数据文件开始分析")

def render_upload():
    st.markdown('<div class="section-title">数据上传</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("选择文件", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file and not st.session_state.data_loaded:
        with st.spinner("加载中..."):
            file_bytes = uploaded_file.getvalue()
            df = load_data(file_bytes, uploaded_file.name)
            if df is not None:
                df, date_cols, num_cols, cat_cols = detect_column_types(df)
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.session_state.file_name = uploaded_file.name
                st.session_state.date_columns = date_cols
                st.session_state.numeric_columns = num_cols
                st.session_state.categorical_columns = cat_cols
                st.session_state.data_loaded = True
                st.rerun()
            else:
                st.error("文件加载失败")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("行数", f"{len(df):,}")
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            st.metric("日期列", len(st.session_state.date_columns))
        with col4:
            st.metric("数值列", len(st.session_state.numeric_columns))

def render_module_selector():
    if not st.session_state.data_loaded:
        return None
    st.markdown('<div class="section-title">功能选择</div>', unsafe_allow_html=True)
    modules = ["数据概览", "异动归因", "交叉分析", "趋势分析", "可视化", "统计分析", "机器学习", "预测分析", "数据清洗"]
    module = st.selectbox("选择功能", modules, label_visibility="collapsed")
    st.session_state.current_module = module
    return module

def render_config(module):
    if not module:
        return None
    st.markdown('<div class="section-title">分析配置</div>', unsafe_allow_html=True)
    df = st.session_state.df
    dims = [c for c in (st.session_state.categorical_columns + st.session_state.numeric_columns) if c not in st.session_state.date_columns]
    metrics = st.session_state.numeric_columns
    dates = st.session_state.date_columns
    
    selected_dims = []
    selected_metrics = []
    date_col = None
    time_range = None
    extra = None
    
    if module in ["异动归因", "交叉分析"]:
        col1, col2 = st.columns(2)
        with col1:
            selected_dims = st.multiselect("维度", dims, default=dims[:1] if dims else [])
        with col2:
            selected_metrics = st.multiselect("指标", metrics, default=metrics[:1] if metrics else [])
        if dates:
            st.markdown("**时间范围**")
            date_col = st.selectbox("日期字段", dates, key="date_col")
            if date_col:
                time_range = render_time_range_selector(df, date_col)
    
    elif module == "趋势分析":
        col1, col2 = st.columns(2)
        with col1:
            selected_dims = st.multiselect("维度", dims, default=dims[:1] if dims else [])
        with col2:
            selected_metrics = st.multiselect("指标", metrics, default=metrics[:1] if metrics else [])
        if dates:
            date_col = st.selectbox("日期字段", dates, key="trend_date")
    
    elif module == "数据概览":
        selected_metrics = st.multiselect("指标", metrics, default=metrics[:4] if metrics else [])
        if dates:
            date_col = st.selectbox("日期字段（可选）", ['无'] + dates, key="overview_date")
            if date_col == '无':
                date_col = None
    
    elif module == "可视化":
        viz_type = st.selectbox("图表类型", ["散点图", "折线图", "柱状图", "箱线图", "热力图"])
        if viz_type == "热力图":
            selected_metrics = st.multiselect("指标", metrics, default=metrics[:4] if metrics else [])
        else:
            col1, col2 = st.columns(2)
            with col1:
                selected_dims = st.multiselect("维度", dims)
            with col2:
                selected_metrics = st.multiselect("指标", metrics, default=metrics[:1] if metrics else [])
        extra = viz_type
    
    elif module == "统计分析":
        selected_metrics = st.multiselect("指标", metrics, default=metrics[:4] if metrics else [])
    
    elif module == "机器学习":
        st.markdown("**算法选择**")
        ml_type = st.selectbox("算法", list(ML_ALGORITHMS.keys()))
        algo_info = ML_ALGORITHMS[ml_type]
        with st.expander(f"关于 {algo_info['name']}"):
            st.markdown(f"<div class='algorithm-card'>{algo_info['desc']}</div>", unsafe_allow_html=True)
        if ml_type == "聚类分析":
            selected_metrics = st.multiselect("特征", metrics, default=metrics[:2] if len(metrics) >= 2 else metrics)
        elif ml_type == "异常检测":
            selected_metrics = st.multiselect("特征", metrics, default=metrics[:2] if len(metrics) >= 2 else metrics)
        elif ml_type == "特征重要性":
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("目标变量", metrics)
            with col2:
                features = st.multiselect("特征变量", [m for m in metrics if m != target], default=[m for m in metrics if m != target][:3])
            selected_metrics = {'target': target, 'features': features}
        elif ml_type == "主成分分析":
            selected_metrics = st.multiselect("特征", metrics, default=metrics[:4] if metrics else [])
        extra = ml_type
    
    elif module == "预测分析":
        if dates:
            date_col = st.selectbox("日期字段", dates, key="forecast_date")
            selected_metrics = st.selectbox("预测指标", metrics)
            periods = st.slider("预测天数", 7, 90, 30)
            selected_metrics = {'metric': selected_metrics, 'periods': periods}
        else:
            st.warning("需要日期字段")
    
    return selected_dims, selected_metrics, date_col, time_range, extra


def render_results(module, dims, metrics, date_col, time_range, extra):
    df = st.session_state.df
    if module == "数据概览":
        render_overview(df, metrics, date_col)
    elif module == "异动归因":
        render_attribution(df, dims, metrics, date_col, time_range)
    elif module == "交叉分析":
        render_cross(df, dims, metrics, date_col, time_range)
    elif module == "趋势分析":
        render_trend(df, dims, metrics, date_col)
    elif module == "可视化":
        render_visualization(df, dims, metrics, extra)
    elif module == "统计分析":
        render_statistics(df, metrics)
    elif module == "机器学习":
        render_ml(df, metrics, extra)
    elif module == "预测分析":
        render_forecast(df, date_col, metrics)
    elif module == "数据清洗":
        render_cleaning(df)

def style_contribution_df(df):
    """样式化归因表格"""
    def color_val(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #dc2626'
            elif val < 0:
                return 'color: #16a34a'
        return ''
    
    styled = df.style.format({
        '基期值': smart_format,
        '目标期值': smart_format,
        '变动': lambda x: f"{x:+,.0f}" if abs(x) >= 100 else f"{x:+.2f}",
        '变动率': '{:+.1f}%',
        '贡献百分比': '{:+.1f}%',
        '贡献pp': lambda x: f"{x:+.2f}pp"
    }).map(color_val, subset=['变动', '变动率', '贡献百分比', '贡献pp'])
    
    return styled

def render_overview(df, metrics, date_col):
    st.markdown('<div class="section-title">数据概览</div>', unsafe_allow_html=True)
    
    # 时间口径选择
    if date_col and date_col in df.columns:
        period = st.selectbox("时间口径", ["天", "周", "月", "年"], key="overview_period")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        if period == "天":
            ts_df = df.groupby(date_col)[metrics].sum().reset_index()
            ts_df = ts_df.sort_values(date_col)
        elif period == "周":
            df['period'] = df[date_col].dt.to_period('W').astype(str)
            ts_df = df.groupby('period')[metrics].sum().reset_index()
        elif period == "月":
            df['period'] = df[date_col].dt.to_period('M').astype(str)
            ts_df = df.groupby('period')[metrics].sum().reset_index()
        elif period == "年":
            df['period'] = df[date_col].dt.to_period('Y').astype(str)
            ts_df = df.groupby('period')[metrics].sum().reset_index()
        
        # 时间线图表
        st.markdown("**时间线**")
        x_col = date_col if period == "天" else 'period'
        
        fig = go.Figure()
        for metric in metrics[:3]:  # 最多显示3个指标
            fig.add_trace(go.Scatter(
                x=ts_df[x_col], y=ts_df[metric],
                mode='lines+markers', name=metric,
                line=dict(width=2)
            ))
        fig.update_layout(
            height=400, plot_bgcolor='white',
            yaxis=dict(gridcolor='#f1f5f9', tickformat=','),
            xaxis=dict(gridcolor='#f1f5f9'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 各维度汇总
        st.markdown("**各维度汇总**")
        dims = [c for c in (st.session_state.categorical_columns + st.session_state.numeric_columns) if c not in st.session_state.date_columns]
        
        if dims and metrics:
            selected_dim = st.selectbox("选择维度", dims, key="overview_dim")
            selected_metric = st.selectbox("选择指标", metrics, key="overview_metric")
            
            dim_summary = df.groupby(selected_dim)[selected_metric].agg(['sum', 'mean', 'count']).reset_index()
            dim_summary.columns = [selected_dim, '总计', '平均', '记录数']
            dim_summary = dim_summary.sort_values('总计', ascending=False)
            
            # 添加总计行
            total_row = pd.DataFrame([{
                selected_dim: '总计',
                '总计': dim_summary['总计'].sum(),
                '平均': df[selected_metric].mean(),
                '记录数': dim_summary['记录数'].sum()
            }])
            dim_summary = pd.concat([total_row, dim_summary], ignore_index=True)
            
            st.dataframe(dim_summary.style.format({'总计': smart_format, '平均': smart_format, '记录数': '{:,}'}), use_container_width=True)
    
    # 数据预览
    st.markdown("**数据预览**")
    st.dataframe(df.head(50), use_container_width=True)

def render_attribution(df, dims, metrics, date_col, time_range):
    st.markdown('<div class="section-title">异动归因</div>', unsafe_allow_html=True)
    
    if not dims or not metrics or not date_col or not time_range or None in time_range:
        st.info("请完成配置")
        return
    
    base_start, base_end, target_start, target_end = time_range
    st.markdown(f"**基期**: {base_start} 至 {base_end} | **目标期**: {target_start} 至 {target_end}")
    
    # 自动对所有维度归因
    st.markdown("---")
    st.markdown("**自动归因（所有维度）**")
    
    all_dims = [c for c in (st.session_state.categorical_columns + st.session_state.numeric_columns) if c not in st.session_state.date_columns]
    
    if st.button("运行自动归因", type="primary"):
        with st.spinner("计算中..."):
            engine = AttributionEngine()
            
            auto_results = []
            for dim in all_dims:
                result_df, total_change, total_base, total_target, total_rate = engine.calculate_contribution(
                    df, dim, metrics[0], date_col, base_start, base_end, target_start, target_end
                )
                if result_df is not None:
                    # 取前3个贡献最大的维度值
                    top3 = result_df[result_df['维度'] != '总计'].head(3)
                    for _, row in top3.iterrows():
                        auto_results.append({
                            '维度': dim,
                            '维度值': row['维度'],
                            '变动': row['变动'],
                            '贡献pp': row['贡献pp']
                        })
            
            auto_df = pd.DataFrame(auto_results)
            auto_df = auto_df.sort_values('贡献pp', key=abs, ascending=False)
            
            def color_pp(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: #dc2626'
                    elif val < 0:
                        return 'color: #16a34a'
                return ''
            
            st.dataframe(auto_df.style.format({'变动': smart_format, '贡献pp': lambda x: f"{x:+.2f}pp"}).map(color_pp, subset=['变动', '贡献pp']), use_container_width=True)
    
    # 单维度详细分析
    st.markdown("---")
    st.markdown("**单维度详细分析**")
    
    selected_dim = st.selectbox("选择维度", dims, key="attr_dim")
    chart_type = st.selectbox("图表类型", ["柱状图", "饼图"], key="attr_chart_type")
    
    if st.button("开始分析", type="primary"):
        with st.spinner("计算中..."):
            engine = AttributionEngine()
            result_df, total_change, total_base, total_target, total_rate = engine.calculate_contribution(
                df, selected_dim, metrics[0], date_col, base_start, base_end, target_start, target_end
            )
            
            if result_df is not None:
                # 总计数据
                st.markdown("**总计**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("基期", smart_format(total_base))
                with col2:
                    st.metric("目标期", smart_format(total_target))
                with col3:
                    st.metric("变动", smart_format(total_change), f"{total_rate:+.1f}%")
                with col4:
                    st.metric("维度数", len(result_df) - 1)
                
                # 基期/目标期对比图（不含总计）
                plot_df = result_df[result_df['维度'] != '总计'].copy()
                
                if chart_type == "柱状图":
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=plot_df['维度'], y=plot_df['基期值'], name='基期', marker_color='#94a3b8'))
                    fig.add_trace(go.Bar(x=plot_df['维度'], y=plot_df['目标期值'], name='目标期', marker_color='#0369a1'))
                    fig.update_layout(barmode='group', title=f"{selected_dim} 基期/目标期对比", height=400)
                else:  # 饼图
                    from plotly.subplots import make_subplots
                    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                       subplot_titles=['基期占比', '目标期占比'])
                    fig.add_trace(go.Pie(labels=plot_df['维度'], values=plot_df['基期值'], name='基期'), 1, 1)
                    fig.add_trace(go.Pie(labels=plot_df['维度'], values=plot_df['目标期值'], name='目标期'), 1, 2)
                    fig.update_layout(title=f"{selected_dim} 占比分析", height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 瀑布图
                fig_waterfall = go.Figure()
                measure = ['absolute']
                x_vals = ['基期总量']
                y_vals = [total_base]
                text_vals = [smart_format(total_base)]
                
                plot_df_sorted = plot_df.sort_values('变动', key=abs, ascending=False)
                for _, row in plot_df_sorted.iterrows():
                    measure.append('relative')
                    x_vals.append(str(row['维度']))
                    y_vals.append(row['变动'])
                    text_vals.append(smart_format(row['变动']) if abs(row['变动']) >= 1 else '')
                
                measure.append('total')
                x_vals.append('目标期总量')
                y_vals.append(total_target)
                text_vals.append(smart_format(total_target))
                
                fig_waterfall.add_trace(go.Waterfall(
                    orientation="v", measure=measure, x=x_vals, y=y_vals, text=text_vals, textposition="outside",
                    connector={"line": {"color": "#e2e8f0", "width": 1}},
                    decreasing={"marker": {"color": "#16a34a"}},
                    increasing={"marker": {"color": "#dc2626"}},
                    totals={"marker": {"color": "#0369a1"}}
                ))
                fig_waterfall.update_layout(title="变动瀑布图", height=450, plot_bgcolor='white', yaxis_tickformat=',')
                st.plotly_chart(fig_waterfall, use_container_width=True)
                
                # 明细表格
                st.markdown("**明细**")
                st.dataframe(style_contribution_df(result_df), use_container_width=True)

def render_cross(df, dims, metrics, date_col, time_range):
    st.markdown('<div class="section-title">交叉分析</div>', unsafe_allow_html=True)
    
    if not dims or not metrics or not date_col or not time_range or None in time_range:
        st.info("请完成配置")
        return
    
    if len(dims) < 2:
        st.info("请选择至少2个维度")
        return
    
    base_start, base_end, target_start, target_end = time_range
    st.markdown(f"**基期**: {base_start} 至 {base_end} | **目标期**: {target_start} 至 {target_end}")
    
    if st.button("开始分析", type="primary"):
        with st.spinner("计算中..."):
            engine = AttributionEngine()
            result_display, result_raw = engine.cross_analysis(
                df, dims[:2], metrics[0], date_col, base_start, base_end, target_start, target_end
            )
            
            if result_display is not None:
                st.dataframe(style_contribution_df(result_display), use_container_width=True)
                
                # 热力图
                try:
                    pivot = result_raw.reset_index().pivot_table(index=dims[0], columns=dims[1], values='变动', fill_value=0)
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot.values, x=pivot.columns, y=pivot.index,
                        text=np.round(pivot.values, 0), texttemplate='%{text}',
                        colorscale=[[0, '#16a34a'], [0.5, '#ffffff'], [1, '#dc2626']], zmid=0
                    ))
                    fig.update_layout(title="变动热力图", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

def render_trend(df, dims, metrics, date_col):
    st.markdown('<div class="section-title">趋势分析</div>', unsafe_allow_html=True)
    
    if not dims or not metrics or not date_col:
        st.info("请完成配置")
        return
    
    if st.button("开始分析", type="primary"):
        with st.spinner("计算中..."):
            engine = AttributionEngine()
            fig, growth_df = engine.trend_analysis(df, dims[0], metrics[0], date_col)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                def color_growth(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return 'color: #dc2626'
                        elif val < 0:
                            return 'color: #16a34a'
                    return ''
                
                styled_growth = growth_df.style.format({
                    '期初值': smart_format, '期末值': smart_format,
                    '变动': lambda x: f"{x:+,.0f}" if abs(x) >= 100 else f"{x:+.2f}",
                    '变动率': '{:+.1f}%',
                    '平均值': smart_format, '最大值': smart_format, '最小值': smart_format
                }).map(color_growth, subset=['变动', '变动率'])
                
                st.dataframe(styled_growth, use_container_width=True)

def render_visualization(df, dims, metrics, viz_type):
    st.markdown('<div class="section-title">可视化</div>', unsafe_allow_html=True)
    
    if viz_type == "散点图" and len(metrics) >= 2:
        fig = px.scatter(df, x=metrics[0], y=metrics[1], color=dims[0] if dims else None)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "折线图" and len(metrics) >= 1:
        fig = px.line(df, x=df.index, y=metrics[0], color=dims[0] if dims else None, markers=True)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "柱状图" and dims and metrics:
        fig = px.bar(df, x=dims[0], y=metrics[0], color=dims[0] if len(dims) > 1 else None)
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "箱线图" and dims and metrics:
        fig = px.box(df, x=dims[0], y=metrics[0])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "热力图" and len(metrics) >= 2:
        corr = df[metrics].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            text=np.round(corr.values, 2), texttemplate='%{text}',
            colorscale=[[0, '#16a34a'], [0.5, '#ffffff'], [1, '#dc2626']], zmid=0, zmin=-1, zmax=1
        ))
        fig.update_layout(title="相关性热力图", height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_statistics(df, metrics):
    st.markdown('<div class="section-title">统计分析</div>', unsafe_allow_html=True)
    
    if not metrics:
        st.info("请选择指标")
        return
    
    stats_module = StatsModule(df)
    
    # 总计
    st.markdown("**总计**")
    cols = st.columns(min(4, len(metrics)))
    for idx, metric in enumerate(metrics[:4]):
        with cols[idx]:
            total = df[metric].sum()
            st.metric(metric, smart_format(total))
    
    tab1, tab2 = st.tabs(["描述统计", "相关性"])
    
    with tab1:
        desc = stats_module.descriptive_stats(metrics)
        st.dataframe(desc.style.format(smart_format), use_container_width=True)
    
    with tab2:
        if len(metrics) >= 2:
            corr, fig = stats_module.correlation(metrics)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr.style.format('{:.3f}'), use_container_width=True)

def render_ml(df, metrics, ml_type):
    st.markdown('<div class="section-title">机器学习</div>', unsafe_allow_html=True)
    
    ml = MLModule(df)
    
    algo_info = ML_ALGORITHMS[ml_type]
    st.markdown(f"**算法**: {algo_info['name']}")
    st.caption(algo_info['desc'])
    
    if ml_type == "聚类分析":
        if not metrics or len(metrics) < 2:
            st.info("请选择至少2个特征")
            return
        n_clusters = st.slider("聚类数", 2, 10, 3)
        if st.button("开始分析", type="primary"):
            with st.spinner("计算中..."):
                fig, stats, score = ml.clustering(metrics, n_clusters)
                if fig:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("轮廓系数", f"{score:.3f}")
                        st.write("聚类中心:")
                        st.dataframe(stats)
    
    elif ml_type == "异常检测":
        if not metrics:
            st.info("请选择特征")
            return
        contamination = st.slider("异常比例", 0.01, 0.3, 0.05)
        if st.button("开始分析", type="primary"):
            with st.spinner("计算中..."):
                fig, count = ml.anomaly_detection(metrics, contamination)
                if fig:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        total = len(df)
                        st.metric("异常数", f"{count} ({count/total*100:.1f}%)")
    
    elif ml_type == "特征重要性":
        target = metrics['target']
        features = metrics['features']
        if not features:
            st.info("请选择特征变量")
            return
        if st.button("开始分析", type="primary"):
            with st.spinner("计算中..."):
                fig, r2, rmse, importance = ml.feature_importance(target, features)
                if fig:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("R²", f"{r2:.3f}")
                        st.metric("RMSE", smart_format(rmse))
                        st.write("重要性排名:")
                        st.dataframe(importance.sort_values('重要性', ascending=False))
    
    elif ml_type == "主成分分析":
        if not metrics or len(metrics) < 2:
            st.info("请选择至少2个特征")
            return
        n_components = st.slider("主成分数", 2, min(5, len(metrics)), 2)
        if st.button("开始分析", type="primary"):
            with st.spinner("计算中..."):
                fig, variance, total_var = ml.pca_analysis(metrics, n_components)
                if fig:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("总解释方差", f"{total_var:.1f}%")
                        st.write("方差比例:")
                        st.dataframe(variance)

def render_forecast(df, date_col, metrics):
    st.markdown('<div class="section-title">预测分析</div>', unsafe_allow_html=True)
    
    if not date_col:
        st.info("需要日期字段")
        return
    
    metric = metrics['metric']
    periods = metrics['periods']
    
    if st.button("开始预测", type="primary"):
        with st.spinner("计算中..."):
            forecast = ForecastModule(df)
            fig, forecast_df, trend = forecast.time_series_forecast(date_col, metric, periods)
            
            if fig:
                hist_avg = df[metric].mean()
                hist_sum = df.groupby(date_col)[metric].sum().mean()
                
                st.markdown("**历史平均**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("日均", smart_format(hist_avg))
                with col2:
                    st.metric("期均", smart_format(hist_sum))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**预测结果**")
                st.dataframe(forecast_df.style.format({'预测值': smart_format}), use_container_width=True)
                
                trend_desc = "上升" if trend > 0 else "下降" if trend < 0 else "平稳"
                st.info(f"趋势: {trend_desc} (日均变化: {smart_format(trend)})")

def render_cleaning(df):
    st.markdown('<div class="section-title">数据清洗</div>', unsafe_allow_html=True)
    
    st.markdown("**数据概况**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总行数", f"{len(df):,}")
    with col2:
        missing = df.isnull().sum().sum()
        st.metric("缺失值", f"{missing:,}")
    with col3:
        dup = df.duplicated().sum()
        st.metric("重复行", f"{dup:,}")
    
    st.markdown("**清洗操作**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("填充数值缺失(均值)", use_container_width=True):
            for col in st.session_state.numeric_columns:
                df[col] = df[col].fillna(df[col].mean())
            st.session_state.df = df
            st.success("已填充")
            st.rerun()
    
    with col2:
        if st.button("填充分类缺失(众数)", use_container_width=True):
            for col in st.session_state.categorical_columns:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            st.session_state.df = df
            st.success("已填充")
            st.rerun()
    
    with col3:
        if st.button("删除重复行", use_container_width=True):
            before = len(df)
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success(f"已删除 {before - len(df)} 行")
            st.rerun()
    
    if st.button("重置数据", use_container_width=True):
        st.session_state.df = st.session_state.df_original.copy()
        st.success("已重置")
        st.rerun()

# ============== 主函数 ==============
def main():
    init_session_state()
    
    with st.sidebar:
        st.markdown("### 数据分析平台")
        st.caption("Data Analytics Platform")
        st.divider()
        
        uploaded_file = st.file_uploader("上传文件", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file and not st.session_state.data_loaded:
            with st.spinner("加载中..."):
                file_bytes = uploaded_file.getvalue()
                df = load_data(file_bytes, uploaded_file.name)
                if df is not None:
                    df, date_cols, num_cols, cat_cols = detect_column_types(df)
                    st.session_state.df = df
                    st.session_state.df_original = df.copy()
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.date_columns = date_cols
                    st.session_state.numeric_columns = num_cols
                    st.session_state.categorical_columns = cat_cols
                    st.session_state.data_loaded = True
                    st.rerun()
        
        if st.session_state.data_loaded:
            st.success(f"已加载: {st.session_state.file_name}")
            st.divider()
            st.markdown("### 功能选择")
            modules = ["数据概览", "异动归因", "交叉分析", "趋势分析", "可视化", "统计分析", "机器学习", "预测分析", "数据清洗"]
            module = st.selectbox("选择功能", modules, label_visibility="collapsed")
            st.session_state.current_module = module
    
    render_welcome()
    
    if st.session_state.data_loaded:
        module = st.session_state.current_module
        st.markdown("---")
        config_result = render_config(module)
        
        if config_result:
            dims, metrics, date_col, time_range, extra = config_result
            st.markdown("---")
            render_results(module, dims, metrics, date_col, time_range, extra)
    else:
        st.info("请在侧边栏上传数据文件")

if __name__ == "__main__":
    main()
