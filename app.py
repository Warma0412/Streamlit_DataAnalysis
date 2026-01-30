import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy import stats
from datetime import datetime, timedelta
import warnings
import io
from itertools import combinations

warnings.filterwarnings('ignore')

pd.set_option("styler.render.max_elements", 1000000)

st.set_page_config(
    page_title="数据分析平台",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f0f9ff; border-right: 1px solid #e0f2fe; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #0369a1; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #bae6fd; }
    .metric-card { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 1px solid #bae6fd; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-value { font-size: 1.5rem; font-weight: 600; color: #0c4a6e; }
    .metric-label { font-size: 0.8rem; color: #64748b; }
    .value-up { color: #dc2626; font-weight: 600; }
    .value-down { color: #16a34a; font-weight: 600; }
    .stButton>button { background-color: #bae6fd; color: #0369a1; border: 1px solid #7dd3fc; border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500; transition: all 0.3s; }
    .stButton>button:hover { background-color: #7dd3fc; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(125, 211, 252, 0.4); }
    .stButton>button[kind="primary"] { background: linear-gradient(135deg, #fbcfe8 0%, #f9a8d4 100%); color: #be185d; border: 1px solid #f9a8d4; }
    .stButton>button[kind="primary"]:hover { background: linear-gradient(135deg, #f9a8d4 0%, #f472b6 100%); }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f8fafc; padding: 4px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; padding: 8px 16px; color: #64748b; transition: all 0.3s; }
    .stTabs [aria-selected="true"] { background-color: #bae6fd !important; color: #0369a1 !important; font-weight: 600; }
    .dataframe { border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }
    th { background-color: #f0f9ff !important; color: #0369a1 !important; font-weight: 600 !important; border-bottom: 2px solid #bae6fd !important; padding: 10px !important; }
    td { border-bottom: 1px solid #f1f5f9 !important; padding: 8px !important; }
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, #e2e8f0, transparent); margin: 1.5rem 0; }
    .algorithm-card { background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); border: 1px solid #fbcfe8; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #f472b6; }
    .algorithm-title { font-weight: 600; color: #be185d; margin-bottom: 0.5rem; }
    .info-box { background: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    .warning-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    .success-box { background: #f0fdf4; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    .dim-tag { display: inline-block; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin: 2px; }
    .metric-tag { display: inline-block; background: #fce7f3; color: #be185d; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin: 2px; }
    .small-note { font-size: 0.75rem; color: #64748b; font-style: italic; }
    .field-container { max-height: 100px; overflow-y: auto; }
    .data-summary { font-size: 0.9rem; color: #475569; }
    .data-summary-value { font-size: 1.1rem; font-weight: 600; color: #0369a1; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        else:
            return pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"加载失败: {e}")
        return None

def init_session_state():
    defaults = {
        'df': None, 'df_original': None, 'file_name': None,
        'date_columns': [], 'numeric_columns': [], 'categorical_columns': [],
        'data_loaded': False, 'current_module': '数据概览',
        'base_start': None, 'base_end': None, 'target_start': None, 'target_end': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def smart_format(val):
    if pd.isna(val) or val is None:
        return "-"
    try:
        num = float(val)
        if abs(num) >= 1000:
            return f"{num:,.0f}"
        elif abs(num) >= 100:
            return f"{num:,.1f}"
        elif abs(num) >= 1:
            return f"{num:,.2f}"
        else:
            return f"{num:,.4f}"
    except:
        return str(val)

def detect_column_types(df):
    date_cols, numeric_cols, cat_cols = [], [], []
    df = df.copy()
    
    for col in df.columns:
        if col.lower() in ['id', 'index', '序号', '编号']:
            continue
            
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
            unique_ratio = df[col].nunique() / len(df)
            if df[col].nunique() < 10 and unique_ratio < 0.05:
                cat_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            try:
                converted = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')
                if converted.notna().sum() / len(df) > 0.8:
                    df[col] = converted
                    unique_ratio = df[col].nunique() / len(df)
                    if df[col].nunique() < 10 and unique_ratio < 0.05:
                        cat_cols.append(col)
                    else:
                        numeric_cols.append(col)
                else:
                    cat_cols.append(col)
            except:
                cat_cols.append(col)
    
    return df, date_cols, numeric_cols, cat_cols

def render_time_range_selector(df, date_col):
    if date_col not in df.columns:
        return None, None, None, None
    
    dates = pd.to_datetime(df[date_col].dropna()).sort_values().unique()
    if len(dates) < 2:
        st.warning("日期数据不足")
        return None, None, None, None
    
    default_target_date = dates[-1]
    default_base_date = dates[-2] if len(dates) >= 2 else dates[-1]
    
    quick_options = st.radio("快捷选择", 
                            ["自定义", "单日对比（昨 vs 今）", "最近7天 vs 前7天", "最近30天 vs 前30天"], 
                            horizontal=True)
    
    if quick_options == "单日对比（昨 vs 今）":
        return (default_base_date.strftime('%Y-%m-%d'), default_base_date.strftime('%Y-%m-%d'),
                default_target_date.strftime('%Y-%m-%d'), default_target_date.strftime('%Y-%m-%d'))
    
    elif quick_options == "最近7天 vs 前7天":
        if len(dates) >= 14:
            base_end = dates[-8]
            base_start = dates[-14]
            target_start = dates[-7]
            target_end = dates[-1]
            return (base_start.strftime('%Y-%m-%d'), base_end.strftime('%Y-%m-%d'),
                    target_start.strftime('%Y-%m-%d'), target_end.strftime('%Y-%m-%d'))
        else:
            st.warning("数据不足14天，使用单日对比")
            return (default_base_date.strftime('%Y-%m-%d'), default_base_date.strftime('%Y-%m-%d'),
                    default_target_date.strftime('%Y-%m-%d'), default_target_date.strftime('%Y-%m-%d'))
    
    elif quick_options == "最近30天 vs 前30天":
        if len(dates) >= 60:
            base_end = dates[-31]
            base_start = dates[-60]
            target_start = dates[-30]
            target_end = dates[-1]
            return (base_start.strftime('%Y-%m-%d'), base_end.strftime('%Y-%m-%d'),
                    target_start.strftime('%Y-%m-%d'), target_end.strftime('%Y-%m-%d'))
        else:
            st.warning("数据不足60天，使用单日对比")
            return (default_base_date.strftime('%Y-%m-%d'), default_base_date.strftime('%Y-%m-%d'),
                    default_target_date.strftime('%Y-%m-%d'), default_target_date.strftime('%Y-%m-%d'))
    
    st.markdown("**基期（对比期）**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        base_year = st.selectbox("年", sorted(set(pd.to_datetime(dates).year), reverse=True), 
                                index=0, key="base_year")
    with col2:
        base_months = sorted(set([d.month for d in dates if d.year == base_year]), reverse=True)
        base_month = st.selectbox("月", base_months, index=0, key="base_month")
    with col3:
        month_dates = [d for d in dates if d.year == base_year and d.month == base_month]
        base_start = st.selectbox("开始日", [d.strftime('%Y-%m-%d') for d in month_dates], 
                                 index=len(month_dates)-1 if month_dates else 0, key="base_start")
    with col4:
        base_end = st.selectbox("结束日", [d.strftime('%Y-%m-%d') for d in month_dates], 
                               index=len(month_dates)-1 if month_dates else 0, key="base_end")
    
    st.markdown("**目标期（分析期）**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_year = st.selectbox("年", sorted(set(pd.to_datetime(dates).year), reverse=True), 
                                  index=0, key="target_year")
    with col2:
        target_months = sorted(set([d.month for d in dates if d.year == target_year]), reverse=True)
        target_month = st.selectbox("月", target_months, index=0, key="target_month")
    with col3:
        month_dates_target = [d for d in dates if d.year == target_year and d.month == target_month]
        target_start = st.selectbox("开始日", [d.strftime('%Y-%m-%d') for d in month_dates_target], 
                                   index=len(month_dates_target)-1 if month_dates_target else 0, key="target_start")
    with col4:
        target_end = st.selectbox("结束日", [d.strftime('%Y-%m-%d') for d in month_dates_target], 
                                 index=len(month_dates_target)-1 if month_dates_target else 0, key="target_end")
    
    return base_start, base_end, target_start, target_end

class AttributionEngine:
    @staticmethod
    def calculate_contribution(df, dimension, metric, date_col, base_start, base_end, target_start, target_end):
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
                
                if base_val != 0:
                    change_rate = (change / base_val) * 100
                else:
                    change_rate = 0
                
                if total_change != 0:
                    contribution_pct = (change / total_change) * 100
                else:
                    contribution_pct = 0
                
                contribution_pp = contribution_pct * total_change_rate / 100
                
                results.append({
                    '维度': dimension,
                    '维度值': dim,
                    '基期值': base_val,
                    '目标期值': target_val,
                    '变动': change,
                    '变动率': change_rate,
                    '贡献百分比': contribution_pct,
                    '贡献pp': contribution_pp
                })
            
            result_df = pd.DataFrame(results).sort_values('变动', key=abs, ascending=False)
            
            total_row = pd.DataFrame([{
                '维度': dimension,
                '维度值': '【总计】',
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
    def multi_dim_analysis(df, dims, metric, date_col, base_start, base_end, target_start, target_end):
        try:
            base_start_dt = pd.to_datetime(base_start)
            base_end_dt = pd.to_datetime(base_end)
            target_start_dt = pd.to_datetime(target_start)
            target_end_dt = pd.to_datetime(target_end)
            df[date_col] = pd.to_datetime(df[date_col])
            
            base_df = df[(df[date_col] >= base_start_dt) & (df[date_col] <= base_end_dt)]
            target_df = df[(df[date_col] >= target_start_dt) & (df[date_col] <= target_end_dt)]
            
            base_data = base_df.groupby(dims)[metric].sum().reset_index()
            target_data = target_df.groupby(dims)[metric].sum().reset_index()
            
            merged = pd.merge(base_data, target_data, on=dims, how='outer', suffixes=('_基期', '_目标期')).fillna(0)
            merged['变动'] = merged[f'{metric}_目标期'] - merged[f'{metric}_基期']
            
            total_base = merged[f'{metric}_基期'].sum()
            total_target = merged[f'{metric}_目标期'].sum()
            total_change = total_target - total_base
            
            merged['变动率'] = np.where(merged[f'{metric}_基期'] != 0, 
                                     (merged['变动'] / merged[f'{metric}_基期']) * 100, 0)
            merged['贡献百分比'] = np.where(total_change != 0, (merged['变动'] / total_change) * 100, 0)
            
            total_change_rate = (total_change / total_base * 100) if total_base != 0 else 0
            merged['贡献pp'] = merged['贡献百分比'] * total_change_rate / 100
            
            merged = merged.rename(columns={
                f'{metric}_基期': '基期值',
                f'{metric}_目标期': '目标期值'
            })
            
            if len(dims) > 2:
                merged['组合维度'] = merged[dims].astype(str).agg(' | '.join, axis=1)
            
            result_cols = dims + ['基期值', '目标期值', '变动', '变动率', '贡献百分比', '贡献pp']
            if len(dims) > 2:
                result_cols.append('组合维度')
            
            merged = merged[result_cols].sort_values('变动', key=abs, ascending=False)
            
            total_row_data = {dim: '【总计】' for dim in dims}
            if len(dims) > 2:
                total_row_data['组合维度'] = '【总计】'
            total_row_data.update({
                '基期值': total_base,
                '目标期值': total_target,
                '变动': total_change,
                '变动率': total_change_rate,
                '贡献百分比': 100.0,
                '贡献pp': total_change_rate
            })
            total_row = pd.DataFrame([total_row_data])
            merged = pd.concat([total_row, merged], ignore_index=True)
            
            return merged, total_change, total_base, total_target
        except Exception as e:
            st.error(f"多维度分析失败: {str(e)}")
            return None, 0, 0, 0

ML_ALGORITHMS = {
    "聚类分析": {
        "KMeans": {"name": "K-Means聚类", "desc": "基于距离的迭代聚类，适合球形分布数据。需指定簇数量。", "params": {"n_clusters": (2, 10, 3)}},
        "DBSCAN": {"name": "DBSCAN密度聚类", "desc": "基于密度的空间聚类，自动识别噪声点，适合不规则形状。", "params": {"eps": (0.1, 2.0, 0.5), "min_samples": (2, 10, 5)}},
        "Hierarchical": {"name": "层次聚类", "desc": "构建树状聚类结构，无需预设簇数量，适合发现层次关系。", "params": {"n_clusters": (2, 10, 3), "linkage": ["ward", "complete", "average"]}}
    },
    "异常检测": {
        "IsolationForest": {"name": "孤立森林", "desc": "基于随机划分的异常检测，对高维数据效果好。", "params": {"contamination": (0.01, 0.3, 0.05)}},
        "ZScore": {"name": "Z-Score统计", "desc": "基于标准差的统计方法，识别偏离均值3倍标准差的异常。", "params": {"threshold": (2, 4, 3)}}
    },
    "回归分析": {
        "LinearRegression": {"name": "线性回归", "desc": "基础的线性关系建模，简单可解释。", "params": {}},
        "Ridge": {"name": "岭回归(L2)", "desc": "添加L2正则化，防止过拟合，适合多重共线性数据。", "params": {"alpha": (0.01, 10.0, 1.0)}},
        "RandomForestRegressor": {"name": "随机森林回归", "desc": "集成多个决策树，处理非线性关系，准确度高。", "params": {"n_estimators": (50, 300, 100), "max_depth": (3, 20, 10)}},
        "GradientBoosting": {"name": "梯度提升回归", "desc": "串行集成学习，精度高，适合竞赛场景。", "params": {"n_estimators": (50, 300, 100), "learning_rate": (0.01, 0.3, 0.1)}},
        "SVR": {"name": "支持向量回归", "desc": "适合非线性回归，通过核函数映射到高维空间。", "params": {"kernel": ["rbf", "linear", "poly"], "C": (0.1, 10.0, 1.0)}},
        "MLPRegressor": {"name": "神经网络回归", "desc": "多层感知器，适合复杂非线性映射。", "params": {"hidden_layer_sizes": [(50,), (100,), (50,50)], "max_iter": (500, 2000, 1000)}}
    },
    "分类分析": {
        "LogisticRegression": {"name": "逻辑回归", "desc": "二分类基础模型，输出概率，可解释性强。", "params": {"C": (0.1, 10.0, 1.0)}},
        "RandomForestClassifier": {"name": "随机森林分类", "desc": "集成决策树，处理特征间复杂交互。", "params": {"n_estimators": (50, 300, 100)}},
        "SVC": {"name": "支持向量机", "desc": "适合高维数据，通过核技巧处理非线性。", "params": {"kernel": ["rbf", "linear"], "C": (0.1, 10.0, 1.0)}},
        "KNN": {"name": "K近邻", "desc": "基于相似度的惰性学习，无需训练过程。", "params": {"n_neighbors": (3, 15, 5)}},
        "DecisionTree": {"name": "决策树", "desc": "树状规则，最易解释，可可视化决策路径。", "params": {"max_depth": (3, 20, 5)}},
        "NaiveBayes": {"name": "朴素贝叶斯", "desc": "基于概率的分类，假设特征独立，速度快。", "params": {}},
        "MLPClassifier": {"name": "神经网络分类", "desc": "多层感知器分类器，适合复杂边界。", "params": {"hidden_layer_sizes": [(50,), (100,)], "max_iter": (500, 2000, 1000)}}
    },
    "降维分析": {
        "PCA": {"name": "主成分分析(PCA)", "desc": "线性降维，保留最大方差方向，去相关性。", "params": {"n_components": (2, 10, 2)}},
        "FeatureImportance": {"name": "特征重要性", "desc": "使用随机森林评估特征对目标的贡献度。", "params": {"n_estimators": (50, 300, 100)}}
    }
}

class AdvancedMLModule:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
    
    def clustering(self, features, algorithm, params):
        try:
            X = self.df[features].dropna()
            if len(X) < 3:
                return None, None, 0, None
            
            X_scaled = self.scaler.fit_transform(X)
            
            if algorithm == "KMeans":
                model = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=params['eps'], min_samples=int(params['min_samples']))
                labels = model.fit_predict(X_scaled)
            elif algorithm == "Hierarchical":
                model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
                labels = model.fit_predict(X_scaled)
            
            X_display = X.copy()
            X_display['聚类'] = labels
            
            mask = labels != -1
            if len(set(labels[mask])) > 1 and mask.sum() > 1:
                score = silhouette_score(X_scaled[mask], labels[mask])
            else:
                score = 0
            
            if len(features) >= 2:
                fig = px.scatter(X_display, x=features[0], y=features[1], color='聚类', 
                               title=f"{algorithm} 聚类结果", color_continuous_scale=px.colors.qualitative.Set1)
            else:
                fig = px.histogram(X_display, x=features[0], color='聚类', 
                                 title=f"{algorithm} 聚类分布")
            fig.update_layout(height=500)
            
            stats = X_display.groupby('聚类')[features].agg(['mean', 'std', 'count']).round(2)
            
            return fig, stats, score, X_display
        except Exception as e:
            st.error(f"聚类失败: {str(e)}")
            return None, None, 0, None

    def anomaly_detection(self, features, algorithm, params):
        try:
            X = self.df[features].dropna()
            if len(X) < 10:
                return None, 0, None
            
            if algorithm == "IsolationForest":
                model = IsolationForest(contamination=params['contamination'], random_state=42)
                y_pred = model.fit_predict(X)
                scores = model.decision_function(X)
            elif algorithm == "ZScore":
                z_scores = np.abs(stats.zscore(X))
                y_pred = np.where((z_scores > params['threshold']).any(axis=1), -1, 1)
                scores = -z_scores.max(axis=1)
            
            X_display = X.copy()
            X_display['类型'] = ['异常' if x == -1 else '正常' for x in y_pred]
            X_display['异常分数'] = scores
            
            if len(features) >= 2:
                fig = px.scatter(X_display, x=features[0], y=features[1], color='类型', 
                               color_discrete_map={'正常': '#0369a1', '异常': '#dc2626'},
                               title="异常检测可视化", size='异常分数' if algorithm == 'IsolationForest' else None)
            else:
                fig = px.histogram(X_display, x=features[0], color='类型',
                                 color_discrete_map={'正常': '#0369a1', '异常': '#dc2626'},
                                 title="异常检测分布")
            fig.update_layout(height=500)
            
            anomaly_count = (y_pred == -1).sum()
            
            return fig, anomaly_count, X_display[['类型', '异常分数'] + features]
        except Exception as e:
            st.error(f"异常检测失败: {str(e)}")
            return None, 0, None

    def regression(self, target, features, algorithm, params, test_size=0.2):
        try:
            df_clean = self.df[features + [target]].dropna()
            if len(df_clean) < 20:
                return None, 0, 0, None, None
            
            X = df_clean[features]
            y = df_clean[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            if algorithm == "LinearRegression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif algorithm == "Ridge":
                model = Ridge(alpha=params['alpha'])
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif algorithm == "RandomForestRegressor":
                model = RandomForestRegressor(n_estimators=int(params['n_estimators']), 
                                            max_depth=int(params['max_depth']), random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif algorithm == "GradientBoosting":
                model = GradientBoostingRegressor(n_estimators=int(params['n_estimators']), 
                                                learning_rate=params['learning_rate'], random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif algorithm == "SVR":
                model = SVR(kernel=params['kernel'], C=params['C'])
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif algorithm == "MLPRegressor":
                model = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'], 
                                   max_iter=int(params['max_iter']), random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='预测值',
                                   marker=dict(color='#0ea5e9', size=8)))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                   mode='lines', name='理想线', line=dict(color='#dc2626', dash='dash')))
            fig.update_layout(title=f"{algorithm} 回归效果 (R²={r2:.3f})", 
                            xaxis_title="实际值", yaxis_title="预测值", height=500)
            
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    '特征': features,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=False)
            elif hasattr(model, 'coef_'):
                importance = pd.DataFrame({
                    '特征': features,
                    '系数': model.coef_
                }).sort_values('系数', key=abs, ascending=False)
            
            return fig, r2, rmse, importance, {
                'MAE': mae, 'RMSE': rmse, 'R²': r2, 
                '样本数': len(df_clean), '训练集': len(X_train), '测试集': len(X_test)
            }
        except Exception as e:
            st.error(f"回归分析失败: {str(e)}")
            return None, 0, 0, None, None

    def classification(self, target, features, algorithm, params, test_size=0.2):
        try:
            df_clean = self.df[features + [target]].dropna()
            if len(df_clean) < 20:
                return None, 0, None, None
            
            X = df_clean[features]
            y = df_clean[target]
            
            if y.dtype in ['float64', 'int64'] and y.nunique() > 5:
                median = y.median()
                y = (y > median).astype(int)
                st.info(f"目标变量已自动二值化（中位数分割: {median:.2f}）")
            
            if y.nunique() < 2:
                st.error("目标变量类别数不足")
                return None, 0, None, None
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            if algorithm == "LogisticRegression":
                model = LogisticRegression(C=params['C'], max_iter=1000)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif algorithm == "RandomForestClassifier":
                model = RandomForestClassifier(n_estimators=int(params['n_estimators']), random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif algorithm == "SVC":
                model = SVC(kernel=params['kernel'], C=params['C'], probability=True)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif algorithm == "KNN":
                model = KNeighborsClassifier(n_neighbors=int(params['n_neighbors']))
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif algorithm == "DecisionTree":
                model = DecisionTreeClassifier(max_depth=int(params['max_depth']), random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif algorithm == "NaiveBayes":
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif algorithm == "MLPClassifier":
                model = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], 
                                    max_iter=int(params['max_iter']), random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(cm, x=[f'预测{i}' for i in sorted(y.unique())], 
                                            y=[f'实际{i}' for i in sorted(y.unique())],
                                            colorscale='Blues', showscale=True)
            fig.update_layout(title=f"{algorithm} 混淆矩阵 (准确率: {acc:.2%})")
            
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    '特征': features,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=False)
            elif hasattr(model, 'coef_'):
                importance = pd.DataFrame({
                    '特征': features,
                    '系数': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                }).sort_values('系数', key=abs, ascending=False)
            
            return fig, acc, report_df, importance, len(y.unique())
        except Exception as e:
            st.error(f"分类分析失败: {str(e)}")
            return None, 0, None, None, 0

    def dimension_reduction(self, features, algorithm, params, target=None):
        try:
            X = self.df[features].dropna()
            if len(X) < 10:
                return None, None, 0, None
            
            X_scaled = self.scaler.fit_transform(X)
            
            if algorithm == "PCA":
                n_comp = min(int(params['n_components']), len(features), len(X))
                model = PCA(n_components=n_comp)
                X_reduced = model.fit_transform(X_scaled)
                
                variance_df = pd.DataFrame({
                    '主成分': [f'PC{i+1}' for i in range(n_comp)],
                    '解释方差比例(%)': model.explained_variance_ratio_ * 100,
                    '累积解释方差(%)': np.cumsum(model.explained_variance_ratio_) * 100
                })
                
                if target and target in self.df.columns:
                    y = self.df.loc[X.index, target]
                    fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1] if n_comp > 1 else np.zeros(len(X_reduced)), 
                                   color=y, title=f"PCA降维结果 ( colored by {target} )")
                else:
                    fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1] if n_comp > 1 else np.zeros(len(X_reduced)),
                                   title="PCA降维结果")
                fig.update_layout(height=500, xaxis_title='PC1', yaxis_title='PC2' if n_comp > 1 else '')
                
                loadings = pd.DataFrame(model.components_.T, columns=[f'PC{i+1}' for i in range(n_comp)], index=features)
                
                return fig, variance_df, model.explained_variance_ratio_.sum() * 100, loadings
                
            elif algorithm == "FeatureImportance":
                if target is None or target not in self.df.columns:
                    st.error("特征重要性需要目标变量")
                    return None, None, 0, None
                
                y = self.df.loc[X.index, target]
                model = RandomForestRegressor(n_estimators=int(params['n_estimators']), random_state=42)
                model.fit(X, y)
                
                importance = pd.DataFrame({
                    '特征': features,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=True)
                
                fig = px.bar(importance, x='重要性', y='特征', orientation='h', title="特征重要性排序")
                fig.update_layout(height=500)
                
                return fig, importance, model.score(X, y), None
                
        except Exception as e:
            st.error(f"降维分析失败: {str(e)}")
            return None, None, 0, None

FORECAST_ALGORITHMS = {
    "Prophet": {
        "name": "Prophet时间序列", 
        "desc": "Facebook开发的加法回归模型，自动处理趋势、季节性和节假日。适合有明显周期性的业务数据。",
        "best_for": "具有强季节性的业务指标",
        "complexity": "中"
    },
    "ARIMA": {
        "name": "ARIMA自回归积分滑动平均", 
        "desc": "经典统计方法，结合自回归和差分。适合平稳或差分后平稳的时间序列。",
        "best_for": "趋势性较强的非季节性数据",
        "complexity": "中"
    },
    "ExponentialSmoothing": {
        "name": "指数平滑法(Holt-Winters)", 
        "desc": "加权平均方法，近期数据权重更高。Holt-Winters增加趋势和季节项。",
        "best_for": "短期预测，平滑趋势数据",
        "complexity": "低"
    },
    "RandomForest_TS": {
        "name": "随机森林(时间序列特征)", 
        "desc": "使用滞后特征、滑动窗口等时序特征工程的机器学习预测。",
        "best_for": "复杂的非线性时间模式",
        "complexity": "高"
    },
    "XGBoost_TS": {
        "name": "XGBoost梯度提升", 
        "desc": "极端梯度提升，使用强大的时间特征工程，通常精度最高。",
        "best_for": "大数据量，复杂模式识别",
        "complexity": "高"
    },
    "LSTM": {
        "name": "LSTM长短期记忆网络", 
        "desc": "深度学习循环神经网络，适合捕捉长期依赖关系。",
        "best_for": "长期模式，大数据量",
        "complexity": "很高"
    }
}

class AdvancedForecastModule:
    def __init__(self, df):
        self.df = df
    
    def prepare_features(self, series, lags=7, window=7):
        """准备时间序列特征，增加空值检查"""
        if len(series) == 0:
            return pd.DataFrame()
            
        df_feat = pd.DataFrame({'y': series})
        df_feat['dayofweek'] = series.index.dayofweek
        df_feat['month'] = series.index.month
        df_feat['day'] = series.index.day
        df_feat['year'] = series.index.year
        
        for i in range(1, lags + 1):
            df_feat[f'lag_{i}'] = series.shift(i)
        
        df_feat[f'rolling_mean_{window}'] = series.rolling(window=window, min_periods=1).mean()
        df_feat[f'rolling_std_{window}'] = series.rolling(window=window, min_periods=1).std()
        df_feat[f'expanding_mean'] = series.expanding(min_periods=1).mean()
        
        df_feat['diff_1'] = series.diff(1)
        df_feat['diff_7'] = series.diff(7)
        
        return df_feat.dropna()
    
    def forecast(self, date_col, metric, algorithm, periods=30, freq='D'):
        try:
            ts_data = self.df.groupby(date_col)[metric].sum().reset_index()
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
            ts_data = ts_data.sort_values(date_col).set_index(date_col)
            
            if len(ts_data) == 0:
                st.error("时间序列数据为空")
                return None, None
                
            ts_data = ts_data.asfreq(freq).fillna(method='ffill')
            
            if len(ts_data) < 30:
                st.warning("时间序列数据少于30天，可能影响预测精度")
            
            # 确保训练数据足够
            if len(ts_data) <= periods:
                st.error(f"数据量({len(ts_data)})不足，需要至少{periods+1}条数据才能进行预测")
                return None, None
                
            train = ts_data.iloc[:-periods]
            
            if len(train) == 0:
                st.error("训练数据为空")
                return None, None
            
            if algorithm == "Prophet":
                return self._prophet_forecast(train, metric, periods, freq)
            elif algorithm == "ARIMA":
                return self._arima_forecast(train, metric, periods)
            elif algorithm == "ExponentialSmoothing":
                return self._exp_smooth_forecast(train, metric, periods)
            elif algorithm == "RandomForest_TS":
                return self._ml_forecast(train, metric, periods, model_type='rf')
            elif algorithm == "XGBoost_TS":
                return self._ml_forecast(train, metric, periods, model_type='xgb')
            elif algorithm == "LSTM":
                return self._lstm_forecast(train, metric, periods)
                
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    def _prophet_forecast(self, train, metric, periods, freq):
        try:
            from prophet import Prophet
        except ImportError:
            st.error("请先安装 Prophet: pip install prophet")
            return None, None
        
        try:
            df_prophet = train.reset_index()
            df_prophet.columns = ['ds', 'y']
            
            # 检查数据有效性
            if len(df_prophet) < 2:
                st.error("Prophet需要至少2个数据点")
                return None, None
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_prophet)
            
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines+markers', 
                                   name='历史数据', line=dict(color='#0369a1')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', 
                                   name='预测值', line=dict(color='#db2777')))
            fig.add_trace(go.Scatter(x=forecast['ds'].tail(periods), y=forecast['yhat'].tail(periods), 
                                   mode='lines', name='预测区间', fill=None, 
                                   line=dict(color='#f472b6', width=0)))
            fig.add_trace(go.Scatter(x=forecast['ds'].tail(periods), 
                                   y=forecast['yhat_upper'].tail(periods), 
                                   mode='lines', fill='tonexty', fillcolor='rgba(244, 114, 182, 0.2)',
                                   line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'].tail(periods), 
                                   y=forecast['yhat_lower'].tail(periods), 
                                   mode='lines', fill='tonexty', fillcolor='rgba(244, 114, 182, 0.2)',
                                   line=dict(width=0), showlegend=False))
            
            fig.update_layout(title=f"{metric} - Prophet预测", height=500, plot_bgcolor='white')
            
            # 安全计算MAPE
            actual = df_prophet['y'].values
            predicted = forecast['yhat'].iloc[:len(df_prophet)].values
            mask = actual != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
            else:
                mape = np.nan
            
            return fig, {
                'MAPE': mape,
                '趋势': '上升' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[-30] else '下降',
                '预测均值': forecast['yhat'].tail(periods).mean(),
                '最后日期': forecast['ds'].max()
            }
        except Exception as e:
            st.error(f"Prophet预测失败: {str(e)}")
            return None, None
    
    def _arima_forecast(self, train, metric, periods):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            st.error("请先安装 statsmodels")
            return None, None
        
        try:
            # 确保数据足够
            if len(train) < 10:
                st.error("ARIMA需要至少10个数据点")
                return None, None
            
            # 简化模型阶数以避免收敛问题
            model = ARIMA(train, order=(2, 1, 1))
            fitted = model.fit()
            
            forecast_result = fitted.get_forecast(steps=periods)
            forecast_mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            future_dates = pd.date_range(start=train.index[-1] + timedelta(days=1), periods=periods, freq=train.index.freq or 'D')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train[metric], mode='lines+markers', 
                                   name='历史数据', line=dict(color='#0369a1')))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_mean, mode='lines', 
                                   name='预测值', line=dict(color='#dc2626')))
            fig.add_trace(go.Scatter(x=future_dates, y=conf_int.iloc[:, 1], mode='lines', 
                                   line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_dates, y=conf_int.iloc[:, 0], mode='lines', 
                                   fill='tonexty', fillcolor='rgba(220, 38, 38, 0.2)',
                                   line=dict(width=0), showlegend=False))
            
            fig.update_layout(title=f"{metric} - ARIMA预测", height=500)
            
            return fig, {
                'AIC': fitted.aic if hasattr(fitted, 'aic') else None,
                'BIC': fitted.bic if hasattr(fitted, 'bic') else None,
                '预测均值': forecast_mean.mean()
            }
        except Exception as e:
            st.error(f"ARIMA预测失败: {str(e)}")
            return None, None
    
    def _exp_smooth_forecast(self, train, metric, periods):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            st.error("请先安装 statsmodels")
            return None, None
        
        try:
            if len(train) < 10:
                st.error("指数平滑需要至少10个数据点")
                return None, None
            
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
            fitted = model.fit()
            
            forecast = fitted.forecast(steps=periods)
            future_dates = pd.date_range(start=train.index[-1] + timedelta(days=1), periods=periods, freq=train.index.freq or 'D')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train[metric], mode='lines+markers', 
                                   name='历史数据', line=dict(color='#0369a1')))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', 
                                   name='预测值', line=dict(color='#059669')))
            
            fig.update_layout(title=f"{metric} - Holt-Winters预测", height=500)
            
            return fig, {
                '平滑水平': fitted.params.get('smoothing_level', None),
                '平滑趋势': fitted.params.get('smoothing_trend', None),
                '平滑季节': fitted.params.get('smoothing_seasonal', None)
            }
        except Exception as e:
            st.error(f"指数平滑预测失败: {str(e)}")
            return None, None
    
    def _ml_forecast(self, train, metric, periods, model_type='rf'):
        """机器学习预测，修复索引越界问题"""
        try:
            # 确保训练数据足够
            if len(train) < 14:
                st.error(f"机器学习预测需要至少14个数据点，当前只有{len(train)}个")
                return None, None
            
            df_features = self.prepare_features(train[metric])
            
            if len(df_features) == 0:
                st.error("特征工程后数据为空")
                return None, None
            
            X = df_features.drop('y', axis=1)
            y = df_features['y']
            
            if len(X) < 10:
                st.error("有效训练样本不足")
                return None, None
            
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            else:
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                except ImportError:
                    st.info("XGBoost未安装，使用RandomForest替代")
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            
            # 安全地获取最后30个值
            last_values_list = train[metric].values
            if len(last_values_list) >= 30:
                last_values = last_values_list[-30:]
            else:
                last_values = last_values_list
            
            if len(last_values) == 0:
                st.error("无法获取历史值进行预测")
                return None, None
            
            predictions = []
            current_values = list(last_values)  # 转为列表方便操作
            
            for i in range(periods):
                # 构建特征
                feat = {
                    'dayofweek': (train.index[-1] + timedelta(days=i+1)).weekday(),
                    'month': (train.index[-1] + timedelta(days=i+1)).month,
                    'day': (train.index[-1] + timedelta(days=i+1)).day,
                    'year': (train.index[-1] + timedelta(days=i+1)).year,
                }
                
                # 安全地获取滞后特征
                for lag in range(1, 8):
                    if len(current_values) >= lag:
                        feat[f'lag_{lag}'] = current_values[-lag]
                    else:
                        feat[f'lag_{lag}'] = current_values[0] if current_values else 0
                
                # 安全地计算滚动统计
                if len(current_values) >= 7:
                    feat['rolling_mean_7'] = np.mean(current_values[-7:])
                    feat['rolling_std_7'] = np.std(current_values[-7:])
                else:
                    feat['rolling_mean_7'] = np.mean(current_values) if current_values else 0
                    feat['rolling_std_7'] = np.std(current_values) if len(current_values) > 1 else 0
                
                feat['expanding_mean'] = np.mean(current_values) if current_values else 0
                
                # 安全地计算差分
                if len(current_values) >= 2:
                    feat['diff_1'] = current_values[-1] - current_values[-2]
                else:
                    feat['diff_1'] = 0
                
                if len(current_values) >= 7:
                    feat['diff_7'] = current_values[-1] - current_values[-7]
                else:
                    feat['diff_7'] = 0
                
                X_pred = pd.DataFrame([feat])
                pred = model.predict(X_pred)[0]
                predictions.append(pred)
                current_values.append(pred)  # 添加到历史值中用于下一个预测
            
            future_dates = pd.date_range(start=train.index[-1] + timedelta(days=1), periods=periods, freq='D')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train[metric], mode='lines', name='历史数据', line=dict(color='#0369a1')))
            fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', 
                                   name='预测值', line=dict(color='#7c3aed')))
            
            fig.update_layout(title=f"{metric} - {model_type.upper()}时序预测", height=500)
            
            importance = pd.DataFrame({
                '特征': X.columns,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            return fig, {
                '模型': model_type,
                '训练集R²': model.score(X, y),
                'Top3特征': importance.head(3)['特征'].tolist(),
                '预测趋势': '上升' if predictions[-1] > predictions[0] else '下降'
            }
        except Exception as e:
            st.error(f"机器学习预测失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    def _lstm_forecast(self, train, metric, periods):
        """LSTM预测，简化版本避免索引问题"""
        st.warning("LSTM预测暂不可用，请使用其他算法")
        return None, None

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
                    '偏度': data.skew(), '峰度': data.kurtosis(),
                    '变异系数': data.std()/data.mean() if data.mean() != 0 else np.nan
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
    
    def normality_test(self, column):
        data = self.df[column].dropna()
        if len(data) < 3:
            return None, None
        stat, p = stats.shapiro(data)
        return stat, p

def create_treemap_figure(df, dims, metric, title):
    plot_df = df[df[dims[0]] != '【总计】'].copy() if '【总计】' in df[dims[0]].values else df.copy()
    
    if len(dims) == 2:
        path = dims
        values = metric
    else:
        path = dims
        values = metric
    
    try:
        fig = px.treemap(plot_df, path=path, values=values, 
                        title=title, color=metric, color_continuous_scale='RdBu')
        fig.update_layout(height=600)
        return fig
    except:
        return None

def create_sunburst_figure(df, dims, metric):
    plot_df = df[df[dims[0]] != '【总计】'].copy() if '【总计】' in df[dims[0]].values else df.copy()
    
    try:
        fig = px.sunburst(plot_df, path=dims, values=metric, color=metric,
                         color_continuous_scale='RdBu')
        fig.update_layout(height=600)
        return fig
    except:
        return None

def render_welcome():
    st.markdown("### 智能数据分析平台")
    st.caption("Advanced Analytics Platform | 上传数据开始智能分析之旅")

def render_upload():
    st.markdown('<div class="section-title">数据上传</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("选择CSV或Excel文件", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file and not st.session_state.data_loaded:
        with st.spinner("正在智能解析数据..."):
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
                st.error("文件加载失败，请检查格式")

def render_data_summary():
    """在功能选择上方展示数据概况（字体减小版）"""
    if not st.session_state.data_loaded:
        return
    
    df = st.session_state.df
    dates = st.session_state.date_columns
    
    n_days = 0
    if dates and dates[0] in df.columns:
        n_days = df[dates[0]].nunique()
    
    n_cols = len(df.columns)
    
    st.markdown('<div class="section-title">数据概况</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='data-summary'>数据天数</div><div class='data-summary-value'>{n_days} 天</div>" if n_days > 0 else "<div class='data-summary'>-</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='data-summary'>字段列数</div><div class='data-summary-value'>{n_cols} 列</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

def render_module_buttons():
    if not st.session_state.data_loaded:
        return None
    
    st.markdown('<div class="section-title">功能选择</div>', unsafe_allow_html=True)
    modules = ["数据概览", "异动归因", "交叉分析", "趋势分析", "可视化", "统计分析", "机器学习", "预测分析", "数据清洗"]
    
    cols = st.columns(3)
    for idx, module in enumerate(modules):
        with cols[idx % 3]:
            btn_type = "primary" if st.session_state.current_module == module else "secondary"
            if st.button(module, key=f"mod_{module}", use_container_width=True, type=btn_type):
                st.session_state.current_module = module
                st.rerun()
    
    return st.session_state.current_module

def render_config(module):
    if not module:
        return None
    st.markdown('<div class="section-title">分析配置</div>', unsafe_allow_html=True)
    df = st.session_state.df
    dims = st.session_state.categorical_columns
    metrics = st.session_state.numeric_columns
    dates = st.session_state.date_columns
    
    selected_dims = []
    selected_metrics = []
    date_col = None
    time_range = None
    extra = None
    
    col_a, col_b = st.columns(2)
    with col_a:
        dim_html = " ".join([f"<span class='dim-tag'>{d}</span>" for d in dims])
        if len(dims) > 15:
            dim_html = " ".join([f"<span class='dim-tag'>{d}</span>" for d in dims[:15]]) + " <span class='dim-tag'>...</span>"
        st.markdown(f"**可用维度:** {dim_html}", unsafe_allow_html=True)
    with col_b:
        metric_html = " ".join([f"<span class='metric-tag'>{m}</span>" for m in metrics])
        if len(metrics) > 15:
            metric_html = " ".join([f"<span class='metric-tag'>{m}</span>" for m in metrics[:15]]) + " <span class='metric-tag'>...</span>"
        st.markdown(f"**可用指标:** {metric_html}", unsafe_allow_html=True)
    
    if dates:
        default_date_idx = 0
    else:
        default_date_idx = None
    
    if module in ["异动归因"]:
        col1, col2 = st.columns(2)
        with col1:
            selected_dims = st.multiselect("分析维度", dims, default=dims[:1] if dims else [])
        with col2:
            selected_metrics = st.multiselect("分析指标", metrics, default=metrics[:1] if metrics else [])
        if dates:
            st.markdown("**时间范围设定**")
            date_col = st.selectbox("日期字段", dates, index=default_date_idx, key="attr_date")
            if date_col:
                time_range = render_time_range_selector(df, date_col)
    
    elif module == "交叉分析":
        col1, col2 = st.columns(2)
        with col1:
            default_dims = dims[:2] if len(dims) >= 2 else dims[:1] if dims else []
            selected_dims = st.multiselect("交叉维度（支持2+维度）", dims, default=default_dims)
            if len(selected_dims) < 2:
                st.warning("请至少选择2个维度进行交叉分析")
        with col2:
            selected_metrics = st.multiselect("分析指标", metrics, default=metrics[:1] if metrics else [])
        if dates:
            date_col = st.selectbox("日期字段", dates, index=default_date_idx, key="cross_date")
            if date_col:
                time_range = render_time_range_selector(df, date_col)
    
    elif module == "趋势分析":
        col1, col2 = st.columns(2)
        with col1:
            selected_dims = st.multiselect("维度（可选，不选看总体）", dims, default=dims[:1] if dims else [])
        with col2:
            selected_metrics = st.multiselect("指标", metrics, default=metrics[:1] if metrics else [])
        if dates:
            date_col = st.selectbox("日期字段", dates, index=default_date_idx, key="trend_date")
    
    elif module == "数据概览":
        if metrics:
            default_metrics = metrics[:min(5, len(metrics))]
            selected_metrics = st.multiselect("展示指标（最多选5个）", metrics, default=default_metrics, max_selections=5)
        if dates:
            date_col = st.selectbox("日期字段（可选）", ['无'] + dates, index=default_date_idx+1 if default_date_idx is not None else 0, key="overview_date")
            if date_col == '无':
                date_col = None
    
    elif module == "可视化":
        viz_type = st.selectbox("图表类型", ["散点图", "折线图", "柱状图", "箱线图", "热力图", "饼图"])
        if viz_type == "热力图":
            selected_metrics = st.multiselect("指标", metrics, default=metrics[:4] if metrics else [])
        elif viz_type == "饼图":
            col1, col2 = st.columns(2)
            with col1:
                selected_dims = st.selectbox("维度", dims)
            with col2:
                selected_metrics = st.selectbox("指标", metrics)
        else:
            col1, col2 = st.columns(2)
            with col1:
                selected_dims = st.multiselect("维度", dims)
            with col2:
                selected_metrics = st.multiselect("指标", metrics, default=metrics[:1] if metrics else [])
        extra = viz_type
    
    elif module == "统计分析":
        selected_metrics = st.multiselect("分析指标", metrics, default=metrics[:4] if metrics else [])
    
    elif module == "机器学习":
        task_type = st.selectbox("任务类型", list(ML_ALGORITHMS.keys()))
        algorithm = st.selectbox("算法选择", list(ML_ALGORITHMS[task_type].keys()))
        
        algo_info = ML_ALGORITHMS[task_type][algorithm]
        st.markdown(f"""
        <div class='algorithm-card'>
            <div class='algorithm-title'>{algo_info['name']}</div>
            <div><b>介绍：</b>{algo_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        params = {}
        for param_name, param_config in algo_info['params'].items():
            if isinstance(param_config, tuple):
                params[param_name] = st.slider(param_name, param_config[0], param_config[1], param_config[2])
            elif isinstance(param_config, list):
                params[param_name] = st.selectbox(param_name, param_config)
        
        if task_type == "降维分析":
            selected_metrics = st.multiselect("特征变量", metrics, default=metrics[:4] if metrics else [])
            if algorithm == "FeatureImportance":
                target = st.selectbox("目标变量", metrics)
                selected_metrics = {"features": selected_metrics, "target": target}
        else:
            col1, col2 = st.columns(2)
            with col1:
                if task_type in ["回归分析", "分类分析", "特征重要性"]:
                    target = st.selectbox("目标变量", metrics)
                    features = st.multiselect("特征变量", [m for m in metrics if m != target], default=[m for m in metrics if m != target][:3])
                    selected_metrics = {"target": target, "features": features}
                else:
                    selected_metrics = st.multiselect("特征变量", metrics, default=metrics[:2] if len(metrics) >= 2 else metrics)
            with col2:
                if task_type in ["聚类分析", "异常检测"]:
                    st.info(f"{algo_info['name']}无需目标变量，将基于特征分布进行分析")
        
        extra = {"task": task_type, "algorithm": algorithm, "params": params}
    
    elif module == "预测分析":
        if dates:
            date_col = st.selectbox("日期字段", dates, index=default_date_idx, key="forecast_date")
            selected_metric = st.selectbox("预测指标", metrics)
            periods = st.slider("预测周期数", 7, 90, 30)
            
            algorithm = st.selectbox("预测算法", list(FORECAST_ALGORITHMS.keys()), 
                                   format_func=lambda x: FORECAST_ALGORITHMS[x]['name'])
            
            algo_info = FORECAST_ALGORITHMS[algorithm]
            st.markdown(f"""
            <div class='info-box'>
                <b>算法特点：</b>{algo_info['desc']}<br>
                <b>适用场景：</b>{algo_info['best_for']}<br>
                <b>复杂度：</b>{algo_info['complexity']}
            </div>
            """, unsafe_allow_html=True)
            
            selected_metrics = {'metric': selected_metric, 'periods': periods, 'algorithm': algorithm}
        else:
            st.warning("需要时间字段进行预测分析")
    
    return selected_dims, selected_metrics, date_col, time_range, extra

def style_contribution_df(df, is_cross=False, dims=None):
    """通用样式函数，仅对贡献pp列进行红绿颜色标注"""
    format_dict = {
        '基期值': smart_format,
        '目标期值': smart_format,
        '变动': smart_format,
        '变动率': '{:.2f}%',
        '贡献百分比': '{:.2f}%'
    }
    
    if '贡献pp' in df.columns:
        format_dict['贡献pp'] = '{:.2f}pp'
    
    # 基础样式（仅格式化，不设置颜色）
    styled = df.style.format(format_dict)
    
    # 只有贡献pp列设置颜色
    if '贡献pp' in df.columns:
        def color_pp_column(col):
            colors = []
            for idx, val in enumerate(col):
                # 判断是否是总计行
                is_total = False
                if is_cross:
                    if '维度值' in df.columns:
                        is_total = df.iloc[idx]['维度值'] == '【总计】'
                    elif '组合维度' in df.columns:
                        is_total = df.iloc[idx]['组合维度'] == '【总计】'
                    elif dims and len(dims) > 0:
                        is_total = str(df.iloc[idx][dims[0]]) == '【总计】'
                    else:
                        is_total = str(df.iloc[idx, 0]) == '【总计】'
                
                # 总计行或非标量值不着色
                if is_total or not isinstance(val, (int, float)):
                    colors.append('')
                else:
                    # 上涨红色，下跌绿色
                    if val > 0:
                        colors.append('color: #dc2626')
                    elif val < 0:
                        colors.append('color: #16a34a')
                    else:
                        colors.append('')
            return colors
        
        styled = styled.apply(color_pp_column, subset=['贡献pp'])
    
    return styled

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

def render_overview(df, metrics, date_col):
    st.markdown('<div class="section-title">数据概览</div>', unsafe_allow_html=True)
    
    if date_col and date_col in df.columns and metrics:
        col1, col2 = st.columns([3, 1])
        with col2:
            period = st.selectbox("时间口径", ["天", "周", "月", "年"], key="overview_period")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        if period == "天":
            ts_df = df.groupby(date_col)[metrics].sum().reset_index()
            ts_df = ts_df.sort_values(date_col)
            x_col = date_col
        elif period == "周":
            df['period'] = df[date_col].dt.to_period('W').astype(str)
            ts_df = df.groupby('period')[metrics].sum().reset_index()
            x_col = 'period'
        elif period == "月":
            df['period'] = df[date_col].dt.to_period('M').astype(str)
            ts_df = df.groupby('period')[metrics].sum().reset_index()
            x_col = 'period'
        elif period == "年":
            df['period'] = df[date_col].dt.to_period('Y').astype(str)
            ts_df = df.groupby('period')[metrics].sum().reset_index()
            x_col = 'period'
        
        st.markdown("**时间线趋势**")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Bold
        for idx, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=ts_df[x_col], y=ts_df[metric],
                mode='lines+markers', name=metric,
                line=dict(color=colors[idx % len(colors)], width=2.5),
                marker=dict(size=6)
            ))
        fig.update_layout(
            height=450, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(gridcolor='#e2e8f0', tickformat=','),
            xaxis=dict(gridcolor='#e2e8f0'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**维度分布分析（最新一天）**")
        st.markdown('<p class="small-note">以下饼图仅基于最新一天的数据分布</p>', unsafe_allow_html=True)
        
        all_dims = st.session_state.categorical_columns
        
        if all_dims and date_col:
            latest_date = df[date_col].max()
            latest_df = df[df[date_col] == latest_date]
            
            st.markdown(f"<p class='small-note'>数据日期：{latest_date.strftime('%Y-%m-%d')} | 该日总记录数：{len(latest_df)}</p>", unsafe_allow_html=True)
            
            display_dims = all_dims[:6]
            dim_cols = st.columns(3)
            
            for idx, dim in enumerate(display_dims):
                with dim_cols[idx % 3]:
                    dim_counts = latest_df[dim].value_counts().head(8)
                    if len(dim_counts) > 0:
                        fig_pie = px.pie(values=dim_counts.values, names=dim_counts.index, 
                                       title=f"{dim} 分布", hole=0.4)
                        fig_pie.update_layout(height=280, showlegend=False, 
                                            margin=dict(l=20, r=20, t=40, b=20),
                                            title_font_size=14)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("**维度详细汇总（最新一天）**")
            selected_dim = st.selectbox("选择维度查看详细统计", all_dims, key="overview_dim_select")
            if metrics:
                selected_metric = st.selectbox("选择汇总指标", metrics, key="overview_metric_select")
                
                dim_summary = latest_df.groupby(selected_dim)[selected_metric].agg(['sum', 'mean', 'count']).reset_index()
                dim_summary.columns = [selected_dim, '总计', '平均', '记录数']
                dim_summary = dim_summary.sort_values('总计', ascending=False)
                
                total_row = pd.DataFrame([{
                    selected_dim: '【总计】',
                    '总计': dim_summary['总计'].sum(),
                    '平均': latest_df[selected_metric].mean(),
                    '记录数': dim_summary['记录数'].sum()
                }])
                dim_summary = pd.concat([total_row, dim_summary], ignore_index=True)
                
                st.dataframe(dim_summary.style.format({'总计': smart_format, '平均': smart_format, '记录数': '{:,}'}), 
                           use_container_width=True)
    
    st.markdown("**数据预览（前50行）**")
    st.dataframe(df.head(50).style.format(smart_format), use_container_width=True)

def render_attribution(df, dims, metrics, date_col, time_range):
    st.markdown('<div class="section-title">异动归因分析</div>', unsafe_allow_html=True)
    
    if not dims or not metrics or not date_col or not time_range or None in time_range:
        st.info("请完成配置并选择时间范围")
        return
    
    base_start, base_end, target_start, target_end = time_range
    st.markdown(f"<div class='info-box'>分析时段 | 基期: {base_start} ~ {base_end} | 目标期: {target_start} ~ {target_end}`</div>", unsafe_allow_html=True)
    
    st.markdown("**全维度自动归因**")
    
    all_dims = st.session_state.categorical_columns
    
    top_n = st.slider("每维度展示Top N", 5, 50, 10, key="auto_attr_top_n")
    
    if st.button("运行全维度归因", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        engine = AttributionEngine()
        
        display_dims = all_dims[:6]
        results_tabs = st.tabs([f"{dim}" for dim in display_dims])
        
        for idx, dim in enumerate(display_dims):
            with results_tabs[idx]:
                with st.spinner(f"分析 {dim}..."):
                    result_df, total_change, total_base, total_target, total_rate = engine.calculate_contribution(
                        df, dim, metrics[0], date_col, base_start, base_end, target_start, target_end
                    )
                    if result_df is not None:
                        total_row = result_df[result_df['维度值'] == '【总计】']
                        detail_rows = result_df[result_df['维度值'] != '【总计】']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("基期", smart_format(total_base))
                        with col2:
                            st.metric("目标期", smart_format(total_target))
                        with col3:
                            st.metric("变动", f"{total_change:+,.2f}", f"{total_rate:.2f}%")
                        with col4:
                            st.metric("维度值数", len(detail_rows))
                        
                        chart_data = detail_rows.head(15)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data['维度值'], y=chart_data['基期值'], 
                                           name='基期', marker_color='#94a3b8', opacity=0.7))
                        fig.add_trace(go.Bar(x=chart_data['维度值'], y=chart_data['目标期值'], 
                                           name='目标期', marker_color='#0ea5e9'))
                        fig.update_layout(barmode='group', height=300, margin=dict(l=20, r=20, t=30, b=80),
                                        xaxis_tickangle=-45, showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        display_df = pd.concat([total_row, detail_rows.head(top_n)], ignore_index=True)
                        st.dataframe(style_contribution_df(display_df, is_cross=False), use_container_width=True, height=300)
            
            progress_bar.progress((idx + 1) / len(display_dims))

def render_cross(df, dims, metrics, date_col, time_range):
    st.markdown('<div class="section-title">交叉分析（支持多维度）</div>', unsafe_allow_html=True)
    
    if not dims or not metrics or not date_col or not time_range or None in time_range:
        st.info("请完成配置并选择至少2个维度和时间范围")
        return
    
    if len(dims) < 2:
        st.warning("请至少选择2个维度进行交叉分析")
        return
    
    base_start, base_end, target_start, target_end = time_range
    st.markdown(f"<div class='info-box'>分析时段 | 基期: {base_start} ~ {base_end} | 目标期: {target_start} ~ {target_end}</div>", unsafe_allow_html=True)
    
    st.markdown(f"**已选维度 ({len(dims)}个):** " + " | ".join([f"**{d}**" for d in dims]))
    
    if st.button("开始交叉分析", type="primary", use_container_width=True):
        with st.spinner("正在进行多维度交叉计算..."):
            engine = AttributionEngine()
            result_display, total_change, total_base, total_target = engine.multi_dim_analysis(
                df, dims, metrics[0], date_col, base_start, base_end, target_start, target_end
            )
            
            if result_display is not None:
                st.markdown(f"<div class='success-box'>总计: {smart_format(total_base)} → {smart_format(total_target)} | 变动: {total_change:+,.2f} ({total_change/total_base*100:.2f}%)</div>", unsafe_allow_html=True)
                
                if len(dims) == 2:
                    tab1, tab2 = st.tabs(["热力图", "明细表"])
                    with tab1:
                        pivot_data = result_display[result_display[dims[0]] != '【总计】']
                        pivot_table = pivot_data.pivot_table(
                            index=dims[0], columns=dims[1], values='变动', fill_value=0
                        )
                        # 红绿配色：负值绿色，正值红色，0为白色
                        fig = px.imshow(
                            pivot_table, 
                            text_auto=True, 
                            aspect="auto", 
                            color_continuous_scale=[(0, "#86efac"), (0.5, "#ffffff"), (1, "#fca5a5")],
                            color_continuous_midpoint=0
                        )
                        fig.update_layout(height=500, title="变动幅度热力图")
                        st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        # 使用交叉分析专用样式，传递dims参数
                        st.dataframe(style_contribution_df(result_display, is_cross=True, dims=dims), use_container_width=True, height=500)
                else:
                    st.markdown("**多维度可视化展示**")
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        fig_tree = create_treemap_figure(result_display, dims, '目标期值', f"{' | '.join(dims)} 层级结构")
                        if fig_tree:
                            st.plotly_chart(fig_tree, use_container_width=True)
                    
                    with viz_col2:
                        fig_sun = create_sunburst_figure(result_display, dims, '目标期值')
                        if fig_sun:
                            st.plotly_chart(fig_sun, use_container_width=True)
                    
                    st.markdown("**层级明细数据**")
                    st.dataframe(style_contribution_df(result_display, is_cross=True, dims=dims), use_container_width=True, height=500)

def render_trend(df, dims, metrics, date_col):
    st.markdown('<div class="section-title">趋势分析</div>', unsafe_allow_html=True)
    
    if not metrics or not date_col:
        st.info("请完成配置")
        return
    
    metric = metrics[0]
    df[date_col] = pd.to_datetime(df[date_col])
    
    if st.button("开始趋势分析", type="primary"):
        with st.spinner("计算趋势..."):
            if dims:
                trend_df = df.groupby([date_col, dims[0]])[metric].sum().reset_index()
                
                fig = px.line(trend_df, x=date_col, y=metric, color=dims[0], 
                            markers=True, title=f"{metric} 分{dims[0]}趋势")
                fig.update_layout(height=500, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                growth_data = []
                for dim_val in trend_df[dims[0]].unique():
                    subdf = trend_df[trend_df[dims[0]] == dim_val].sort_values(date_col)
                    if len(subdf) >= 2:
                        first, last = subdf[metric].iloc[0], subdf[metric].iloc[-1]
                        growth_data.append({
                            dims[0]: dim_val,
                            '首值': first, '末值': last,
                            '变动': last - first,
                            '增长率%': (last-first)/first*100 if first != 0 else 0,
                            '均值': subdf[metric].mean(),
                            '趋势': '上升' if last > first else '下降' if last < first else '平稳'
                        })
                
                growth_df = pd.DataFrame(growth_data).sort_values('变动', key=abs, ascending=False)
                
                styled_growth = growth_df.style.format({
                    '首值': smart_format, 
                    '末值': smart_format,
                    '变动': lambda x: f"{x:+,.2f}",
                    '增长率%': '{:.2f}%', 
                    '均值': smart_format
                }).map(lambda x: 'color: #dc2626' if x > 0 else 'color: #16a34a' if x < 0 else '', 
                       subset=['变动', '增长率%'])
                
                st.dataframe(styled_growth, use_container_width=True)
            else:
                total_trend = df.groupby(date_col)[metric].sum().reset_index()
                fig = px.line(total_trend, x=date_col, y=metric, markers=True, title=f"{metric} 整体趋势")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

def render_visualization(df, dims, metrics, viz_type):
    st.markdown('<div class="section-title">可视化</div>', unsafe_allow_html=True)
    
    # 移除抽样，使用全量数据
    df_full = df
    
    if viz_type == "散点图" and len(metrics) >= 2:
        fig = px.scatter(df_full, x=metrics[0], y=metrics[1], color=dims[0] if dims else None,
                        trendline="ols", title=f"{metrics[0]} vs {metrics[1]}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "折线图" and len(metrics) >= 1:
        fig = px.line(df_full, x=df_full.index, y=metrics[0], color=dims[0] if dims else None, 
                     markers=True, title=f"{metrics[0]} 走势")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "柱状图" and dims and metrics:
        agg_df = df_full.groupby(dims[0])[metrics[0]].sum().reset_index().sort_values(metrics[0], ascending=False).head(20)
        fig = px.bar(agg_df, x=dims[0], y=metrics[0], color=metrics[0], 
                    color_continuous_scale='Blues', title=f"Top 20 {dims[0]}")
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "箱线图" and dims and metrics:
        fig = px.box(df_full, x=dims[0], y=metrics[0], title=f"{metrics[0]} 分布（按{dims[0]}）")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "饼图" and dims and metrics:
        agg_df = df_full.groupby(dims)[metrics].sum().reset_index().sort_values(metrics, ascending=False).head(10)
        fig = px.pie(agg_df, names=dims, values=metrics, title=f"{dims} 占比分析", hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "热力图" and len(metrics) >= 2:
        corr = df_full[metrics].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                       title="相关性矩阵")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def render_statistics(df, metrics):
    st.markdown('<div class="section-title">统计分析</div>', unsafe_allow_html=True)
    
    if not metrics:
        st.info("请选择分析指标")
        return
    
    # 移除指标数量限制，使用全部选中指标
    display_metrics = metrics
    
    stats_module = StatsModule(df)
    
    tab1, tab2, tab3 = st.tabs(["描述统计", "相关性分析", "正态性检验"])
    
    with tab1:
        desc = stats_module.descriptive_stats(display_metrics)
        st.dataframe(desc.style.format(smart_format), use_container_width=True)
        
        # 限制箱线图数量避免过长
        display_for_box = display_metrics[:10] if len(display_metrics) > 10 else display_metrics
        if len(display_metrics) > 10:
            st.caption(f"指标较多，箱线图仅展示前10个")
        
        fig = go.Figure()
        for metric in display_for_box:
            fig.add_trace(go.Box(y=df[metric], name=metric))
        fig.update_layout(title="箱线图分布", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if len(display_metrics) >= 2:
            corr, fig = stats_module.correlation(display_metrics)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**相关性说明：**")
            st.markdown("- |r| > 0.8: 强相关")
            st.markdown("- 0.5 < |r| < 0.8: 中等相关")
            st.markdown("- |r| < 0.3: 弱相关")
    
    with tab3:
        results = []
        for col in display_metrics:
            stat, p = stats_module.normality_test(col)
            if stat is not None:
                results.append({
                    '指标': col,
                    'W统计量': stat,
                    'P值': p,
                    '是否正态': '是' if p > 0.05 else '否',
                    '解释': '数据近似正态分布' if p > 0.05 else '数据偏离正态分布'
                })
        if results:
            st.dataframe(pd.DataFrame(results).style.format({
                'W统计量': '{:.4f}',
                'P值': '{:.4f}'
            }), use_container_width=True)

def render_ml(df, metrics, extra):
    st.markdown('<div class="section-title">机器学习分析</div>', unsafe_allow_html=True)
    
    ml = AdvancedMLModule(df)
    task = extra['task']
    algorithm = extra['algorithm']
    params = extra['params']
    
    if st.button("开始训练", type="primary", use_container_width=True):
        progress_placeholder = st.empty()
        progress_placeholder.info("正在训练模型...")
        
        if task == "聚类分析":
            features = metrics if isinstance(metrics, list) else metrics['features']
            if not features or len(features) < 2:
                st.error("请至少选择2个特征")
                return
            fig, stats, score, labeled_data = ml.clustering(features, algorithm, params)
            if fig:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("轮廓系数", f"{score:.3f}", help="越接近1越好")
                    if algorithm == "KMeans":
                        st.metric("簇数量", params['n_clusters'])
                    elif algorithm == "DBSCAN":
                        n_clusters = len(set(stats.index)) - (1 if -1 in stats.index else 0)
                        st.metric("识别簇数", n_clusters)
                    st.markdown("**各簇统计:**")
                    st.dataframe(stats, use_container_width=True)
        
        elif task == "异常检测":
            features = metrics if isinstance(metrics, list) else metrics['features']
            fig, count, details = ml.anomaly_detection(features, algorithm, params)
            if fig:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    total = len(df)
                    pct = count/total*100
                    st.metric("异常样本数", count)
                    st.metric("异常比例", f"{pct:.2f}%")
                    st.warning(f"发现 {count} 个异常值") if count > 0 else st.success("未发现明显异常")
        
        elif task == "回归分析":
            target = metrics['target']
            features = metrics['features']
            if not features:
                st.error("请选择特征变量")
                return
            fig, r2, rmse, importance, metrics_dict = ml.regression(target, features, algorithm, params)
            if fig:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("R² 决定系数", f"{r2:.4f}", help="越接近1越好")
                    st.metric("RMSE", smart_format(rmse), help="均方根误差")
                    st.metric("MAE", smart_format(metrics_dict['MAE']), help="平均绝对误差")
                    if importance is not None:
                        st.markdown("**特征重要性:**")
                        st.dataframe(importance.head(), use_container_width=True)
        
        elif task == "分类分析":
            target = metrics['target']
            features = metrics['features']
            fig, acc, report, importance, n_classes = ml.classification(target, features, algorithm, params)
            if fig:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("准确率", f"{acc:.2%}")
                    st.markdown("**分类报告:**")
                    st.dataframe(report.style.format("{:.3f}"), use_container_width=True, height=300)
        
        elif task == "降维分析":
            if algorithm == "PCA":
                features = metrics if isinstance(metrics, list) else metrics['features']
                fig, variance, total_var, loadings = ml.dimension_reduction(features, algorithm, params)
                if fig:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("累积解释方差", f"{total_var:.1f}%")
                        st.markdown("**方差解释:**")
                        st.dataframe(variance, use_container_width=True)
                        st.markdown("**成分载荷:**")
                        st.dataframe(loadings.round(3), use_container_width=True, height=250)
            elif algorithm == "FeatureImportance":
                target = metrics['target']
                features = metrics['features']
                fig, importance, score, _ = ml.dimension_reduction(features, algorithm, params, target)
                if fig:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("模型R²", f"{score:.4f}")
                        st.markdown("**重要性排序:**")
                        st.dataframe(importance.sort_values('重要性', ascending=False), use_container_width=True)
        
        progress_placeholder.success("分析完成！")

def render_forecast(df, date_col, metrics):
    st.markdown('<div class="section-title">智能预测分析</div>', unsafe_allow_html=True)
    
    if not date_col:
        st.info("需要时间字段")
        return
    
    metric = metrics['metric']
    periods = metrics['periods']
    algorithm = metrics['algorithm']
    
    if st.button("开始预测", type="primary", use_container_width=True):
        with st.spinner(f"正在使用 {FORECAST_ALGORITHMS[algorithm]['name']} 进行预测..."):
            forecast_module = AdvancedForecastModule(df)
            fig, info = forecast_module.forecast(date_col, metric, algorithm, periods)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                if info:
                    st.markdown("**预测评估**")
                    cols = st.columns(len(info))
                    for idx, (key, value) in enumerate(info.items()):
                        with cols[idx]:
                            if isinstance(value, (int, float)):
                                st.metric(key, f"{value:.4f}" if isinstance(value, float) else f"{value:,}")
                            else:
                                st.metric(key, str(value))

def render_cleaning(df):
    st.markdown('<div class="section-title">数据清洗</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        missing = df.isnull().sum().sum()
        st.metric("缺失值总数", f"{ missing :,}")
    with col2:
        dup = df.duplicated().sum()
        st.metric("重复行数", f"{dup:,}")
    with col3:
        st.metric("内存占用", f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    
    st.markdown("**清洗操作**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("填充数值缺失(均值)", use_container_width=True):
            for col in st.session_state.numeric_columns:
                df[col] = df[col].fillna(df[col].mean())
            st.session_state.df = df
            st.success("数值缺失已填充")
            st.rerun()
    
    with col2:
        if st.button("填充分类缺失(众数)", use_container_width=True):
            for col in st.session_state.categorical_columns:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            st.session_state.df = df
            st.success("分类缺失已填充")
            st.rerun()
    
    with col3:
        if st.button("删除重复行", use_container_width=True):
            before = len(df)
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success(f"已删除 {before - len(df)} 行重复数据")
            st.rerun()
    
    with col4:
        if st.button("重置所有数据", use_container_width=True):
            st.session_state.df = st.session_state.df_original.copy()
            st.success("数据已重置为原始状态")
            st.rerun()

def main():
    init_session_state()
    
    with st.sidebar:
        st.markdown("### 🐧数据分析平台")
        st.caption("Advanced Analytics Platform")
        st.divider()
        
        render_upload()
        
        if st.session_state.data_loaded:
            render_data_summary()
            st.divider()
            render_module_buttons()
    
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
        st.info("请在左侧上传数据文件开始分析喵~")

if __name__ == "__main__":
    main()
