
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import joblib
# import pandas as pd
# from matplotlib.ticker import FormatStrFormatter
# from sklearn.neural_network import MLPRegressor
# import warnings
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from sklearn.metrics import mean_squared_error
# warnings.filterwarnings('ignore')

# # 设置随机种子
# torch.manual_seed(42)
# np.random.seed(42)

# # 设置页面配置
# st.set_page_config(
#     page_title="Vehicle Impact Force Prediction System",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # 设置matplotlib字体 - 改回Times New Roman
# plt.rcParams.update({
#     'font.family': 'Times New Roman',
#     'font.size': 10,
#     'axes.linewidth': 0.8,
#     'axes.edgecolor': 'black',
#     'figure.frameon': True,
#     'figure.dpi': 150,
# })


# class HybridBridgeImpactDataset(Dataset):
#     """混合模型数据集"""
#     def __init__(self, mat_path, normalize=True):
#         # 加载数据
#         data = sio.loadmat(mat_path)
        
#         # 获取数据 
#         # 注意：新数据集vehicle_and_bridge_collision_dataset.mat只有1条样本
#         self.force_data = data['force_data']  # (1, 601)
#         self.response_data = data['response_data']  # (1, 601)
#         self.force_fft_real = data['force_fft_real']  # (1, 150)
#         self.force_fft_imag = data['force_fft_imag']  # (1, 150)
#         self.freq_vector = data['freq_vector']  # (1, 150)
#         self.time_vector = data['time_vector']  # (1, 601)
#         self.sampling_rate = float(data['sampling_rate'])
        
#         # 转换为numpy数组（与训练代码一致）
#         self.force_data = np.array(self.force_data)
#         self.response_data = np.array(self.response_data)
#         self.force_fft_real = np.array(self.force_fft_real)
#         self.force_fft_imag = np.array(self.force_fft_imag)
#         self.freq_vector = np.array(self.freq_vector)
#         self.time_vector = np.array(self.time_vector)
        
#         # 数据形状
#         self.n_samples = self.force_data.shape[0]  # 现在为1
#         self.n_time = self.force_data.shape[1]  # 601
#         self.n_freq = self.force_fft_real.shape[1]  # 150
        
#         # 提取时域和频域特征
#         self.extract_features()
        
#         # 准备训练数据
#         self.prepare_expanded_dataset(normalize)
    
#     def extract_features(self):
#         """提取时域和频域特征"""
#         # 时域特征
#         time_features = []
#         for i in range(self.n_samples):
#             response = self.response_data[i]
            
#             # 基本统计特征
#             mean_val = response.mean()
#             std_val = response.std()
#             min_val = response.min()
            
#             # 峰值特征
#             max_val = response.max()
#             rms_val = np.sqrt(np.mean(response**2))
#             peak_to_peak = max_val - min_val
#             crest_factor = max_val / rms_val if rms_val != 0 else 0
#             shape_factor = rms_val / np.mean(np.abs(response)) if np.mean(np.abs(response)) != 0 else 0
            
#             time_features.append([
#                 mean_val, std_val, min_val,  # 保留的3个基本特征
#                 peak_to_peak, crest_factor, shape_factor  # 保留的3个峰值特征
#             ])
        
#         self.time_features = np.array(time_features)  # (n_samples, 6)
        
#         # 频域特征
#         freq_features = []
#         for i in range(self.n_samples):
#             response = self.response_data[i]
#             fft_result = np.fft.fft(response)[:self.n_freq]  # 取前n_freq个频率
            
#             # 频域统计特征
#             mag_spectrum = np.abs(fft_result)
            
#             # 基本频域统计
#             freq_mean = mag_spectrum.mean()
#             freq_std = mag_spectrum.std()
#             freq_max = mag_spectrum.max()
#             freq_min = mag_spectrum.min()
            
#             # 频域形状特征
#             freq_skewness = self._skewness(mag_spectrum)
#             freq_kurtosis = self._kurtosis(mag_spectrum)
            
#             sample_features = [
#                 freq_mean, freq_std, freq_min, freq_max,  # 保留的4个基本统计特征
#                 freq_skewness, freq_kurtosis  # 保留的2个形状特征
#             ]
            
#             freq_features.append(sample_features)
        
#         self.freq_features = np.array(freq_features)  # (n_samples, 6)
    
#     def _skewness(self, x):
#         """计算偏度"""
#         n = len(x)
#         if n < 2:
#             return 0
#         mean = np.mean(x)
#         std = np.std(x) + 1e-8
#         skew = np.mean((x - mean)**3) / (std**3)
#         return skew
    
#     def _kurtosis(self, x):
#         """计算峰度"""
#         n = len(x)
#         if n < 2:
#             return 0
#         mean = np.mean(x)
#         std = np.std(x) + 1e-8
#         kurt = np.mean((x - mean)**4) / (std**4) - 3  # 超额峰度
#         return kurt
    
#     def prepare_expanded_dataset(self, normalize=True):
#         """准备扩展的数据集 """
#         # 扩展响应数据：每个样本重复n_freq次
#         response_expanded = np.repeat(self.response_data, self.n_freq, axis=0)  # (n_samples*n_freq, 601)
        
#         # 扩展时域特征
#         time_features_expanded = np.repeat(self.time_features, self.n_freq, axis=0)  # (n_samples*n_freq, 6)
        
#         # 扩展频域特征
#         freq_features_expanded = np.repeat(self.freq_features, self.n_freq, axis=0)  # (n_samples*n_freq, 6)
        
#         # 扩展频率：为每个样本的每个频率创建频率向量
#         freq_expanded = np.tile(self.freq_vector.T, (self.n_samples, 1)).reshape(-1, 1)  # (n_samples*n_freq, 1)
        
#         # 扩展FFT实部和虚部
#         real_expanded = self.force_fft_real.reshape(-1, 1)  # (n_samples*n_freq, 1)
#         imag_expanded = self.force_fft_imag.reshape(-1, 1)  # (n_samples*n_freq, 1)
        
#         # 组合所有特征
#         combined_features = np.hstack([
#             response_expanded,  # 原始时程数据 (601)
#             time_features_expanded,  # 时域特征 (6)
#             freq_features_expanded,  # 频域特征 (6)
#             freq_expanded  # 频率 (1)
#         ])  # 总共 614 维特征
        
#         # 归一化
#         if normalize:
#             # 归一化特征            
#             # 加载归一化参数文件中的均值和标准差
#             norm_params = sio.loadmat('normalization_parameters.mat')
#             self.feature_mean = norm_params['feature_mean']
#             self.feature_std = norm_params['feature_std']           
#             self.real_mean = norm_params['real_mean'].item()
#             self.real_std = norm_params['real_std'].item()
#             self.imag_mean = norm_params['imag_mean'].item()
#             self.imag_std = norm_params['imag_std'].item()
            
#             # 使用从文件加载的参数进行归一化
#             self.features_normalized = (combined_features - self.feature_mean) / (self.feature_std + 1e-8)
#             self.real_normalized = (real_expanded - self.real_mean) / (self.real_std + 1e-8)
#             self.imag_normalized = (imag_expanded - self.imag_mean) / (self.imag_std + 1e-8)
#         else:
#             self.features_normalized = combined_features
#             self.real_normalized = real_expanded
#             self.imag_normalized = imag_expanded
        
#         # 存储原始数据用于重建
#         self.response_expanded = response_expanded
#         self.freq_expanded = freq_expanded
#         self.real_expanded = real_expanded
#         self.imag_expanded = imag_expanded
        
#         self.expanded_size = len(self.features_normalized)
        
#         # 转换为torch张量以保持与原始代码兼容
#         self.features_normalized = torch.FloatTensor(self.features_normalized)
#         self.real_normalized = torch.FloatTensor(self.real_normalized)
#         self.imag_normalized = torch.FloatTensor(self.imag_normalized)
#         self.response_expanded = torch.FloatTensor(self.response_expanded)
#         self.freq_expanded = torch.FloatTensor(self.freq_expanded)
#         self.real_expanded = torch.FloatTensor(self.real_expanded)
#         self.imag_expanded = torch.FloatTensor(self.imag_expanded)
    
#     def __len__(self):
#         return self.expanded_size
    
#     def __getitem__(self, idx):
#         return {
#             'features': self.features_normalized[idx],
#             'real': self.real_normalized[idx],
#             'imag': self.imag_normalized[idx],
#             'response_original': self.response_expanded[idx],
#             'freq_original': self.freq_expanded[idx],
#             'real_original': self.real_expanded[idx],
#             'imag_original': self.imag_expanded[idx]
#         }

# class CustomMLPRegressor(MLPRegressor):
#     """自定义MLP回归器，用于处理字符串表示的隐藏层结构"""
#     def set_params(self, **params):
#         # 转换字符串表示的隐藏层结构
#         if 'hidden_layer_sizes_str' in params:
#             hidden_str = params.pop('hidden_layer_sizes_str')
#             # 将字符串转换为元组
#             if '_' in hidden_str:
#                 params['hidden_layer_sizes'] = tuple(int(x) for x in hidden_str.split('_'))
#             else:
#                 params['hidden_layer_sizes'] = int(hidden_str)
#         return super().set_params(**params)

# class HybridModel:
#     """机器学习模型"""
#     def __init__(self):
#         self.ml_models = {'real': [], 'imag': []}
#         self.meta_models = {'real': None, 'imag': None}
        
#     def load_models(self, ml_model_path, meta_model_path):
#         """加载已训练的模型"""
#         # 加载机器学习模型
#         ml_models_data = joblib.load(ml_model_path)
        
#         # 重新构建ml_models结构
#         for name, model_list in ml_models_data['real']:
#             self.ml_models['real'].append((name, model_list))
        
#         for name, model_list in ml_models_data['imag']:
#             self.ml_models['imag'].append((name, model_list))
        
#         # 加载元模型
#         self.meta_models = joblib.load(meta_model_path)
        
#         return self
    
#     def predict(self, X):
#         """使用混合模型进行预测"""
#         # 获取基模型预测
#         real_base_preds = []
#         imag_base_preds = []
        
#         # 机器学习模型预测
#         for name, model_list in self.ml_models['real']:
#             model = model_list[0]  # 获取单个模型
#             predictions = model.predict(X)
#             real_base_preds.append(predictions.reshape(-1, 1))
        
#         for name, model_list in self.ml_models['imag']:
#             model = model_list[0]  # 获取单个模型
#             predictions = model.predict(X)
#             imag_base_preds.append(predictions.reshape(-1, 1))
        
#         # 合并所有预测
#         real_features = np.hstack(real_base_preds)
#         imag_features = np.hstack(imag_base_preds)
        
#         # 元模型预测
#         real_pred_final = self.meta_models['real'].predict(real_features)
#         imag_pred_final = self.meta_models['imag'].predict(imag_features)
        
#         return real_pred_final, imag_pred_final

# # 初始化session state中的缩放因子
# if 'response_zoom' not in st.session_state:
#     st.session_state.response_zoom = 1.0
# if 'impact_zoom' not in st.session_state:
#     st.session_state.impact_zoom = 1.0

# # 主应用程序
# def main():
#     # 创建标题栏布局，将标题和校徽放在同一行
#     col_title, col_logo = st.columns([3, 1])
    
#     with col_title:
#         # 标题左对齐，设置深蓝色（RGB: 0, 51, 102）
#         st.markdown("<h1 style='text-align: left; color: rgb(0, 51, 102); margin-bottom: 3px;'>🚗 Vehicle Impact Force Prediction System for Bridge Piers</h1>", unsafe_allow_html=True)
    
#     with col_logo:
#         # 校徽右对齐
#         try:
#             # 创建一个右对齐的容器
#             st.markdown(
#                 """
#                 <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#             st.image("seu1_logo.png", width=250)
#         except:
#             st.markdown("<div style='display: flex; justify-content: flex-end;'><strong>SEU Logo</strong></div>", unsafe_allow_html=True)
    
#     # 添加与标题颜色一致的横线，加粗并靠近标题
#     st.markdown("<hr style='height:4px; border-width:0; color:rgb(0, 51, 102); background-color:rgb(0, 51, 102); margin-top: 0; margin-bottom: 18px;'>", unsafe_allow_html=True)
    
#     # 开发者信息 - 左对齐（现在放在第二行）
#     st.markdown("<h4 style='text-align: left; margin-bottom: 0;'>Developed by B.Q. Wang, D. Feng, et al.</h4>", unsafe_allow_html=True)
#     st.markdown("<h4 style='text-align: left; margin-top: 0;'>Southeast University (SEU), Nanjing, Jiangsu, China</h4>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: left;'>E-mail: bqwang@seu.edu.cn</p>", unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # 初始化session state
#     if 'data_loaded' not in st.session_state:
#         st.session_state.data_loaded = False
#     if 'model_loaded' not in st.session_state:
#         st.session_state.model_loaded = False
#     if 'dataset' not in st.session_state:
#         st.session_state.dataset = None
#     if 'hybrid_model' not in st.session_state:
#         st.session_state.hybrid_model = None
#     if 'predictions' not in st.session_state:
#         st.session_state.predictions = None
#     if 'prediction_made' not in st.session_state:
#         st.session_state.prediction_made = False
    
#     # 侧边栏
#     with st.sidebar:
#         st.header("📊 Data Control")
        
#         # 数据加载部分
#         st.subheader("1. Load Data")
#         data_path = st.text_input("Data file path", "vehicle_and_bridge_collision_dataset.mat")
        
#         if st.button("📂 Load Data", use_container_width=True):
#             with st.spinner("Loading data..."):
#                 try:
#                     dataset = HybridBridgeImpactDataset(data_path, normalize=True)  
#                     st.session_state.dataset = dataset
#                     st.session_state.data_loaded = True
#                     st.success(f"Data loaded successfully! {dataset.n_samples} sample in total")
#                 except Exception as e:
#                     st.error(f"Failed to load data: {str(e)}")
        
#         if st.session_state.data_loaded:
#             st.info(f"✅ Data loaded")
            
#             # 模型加载部分
#             st.subheader("2. Load Model")
#             ml_model_path = st.text_input("Base model path", "hybrid_ml_models.joblib")
#             meta_model_path = st.text_input("Meta model path", "hybrid_meta_models.joblib")
            
#             if st.button("🤖 Load Model", use_container_width=True):
#                 with st.spinner("Loading model..."):
#                     try:
#                         hybrid_model = HybridModel()
#                         hybrid_model.load_models(ml_model_path, meta_model_path)
#                         st.session_state.hybrid_model = hybrid_model
#                         st.session_state.model_loaded = True
#                         st.success("Model loaded successfully!")
#                     except Exception as e:
#                         st.error(f"Failed to load model: {str(e)}")
        
#         if st.session_state.model_loaded:
#             st.info("✅ Model loaded")
            
#             # 预测按钮
#             if st.session_state.data_loaded:
#                 st.subheader("3. Make Prediction")
#                 if st.button("🔮 Predict Impact Force", use_container_width=True, type="primary"):
#                     st.session_state.prediction_made = True
#                     st.rerun()
                
#                 # 添加Plot Controls组件
#                 st.subheader("🔍 Plot Controls")
#                 col1, col2, col3 = st.columns(3)

#                 with col1:
#                     # 使用不可见字符强制三行布局
#                     button_text = """➕
#                 Zoom
#                 in"""
#                     if st.button(button_text, use_container_width=True):
#                         st.session_state.response_zoom *= 0.8
#                         st.session_state.impact_zoom *= 0.8
#                         st.rerun()

#                 with col2:
#                     button_text = """➖
#                 Zoom
#                 out"""
#                     if st.button(button_text, use_container_width=True):
#                           st.session_state.response_zoom *= 1.25
#                           st.session_state.impact_zoom *= 1.25
#                           st.rerun()

#                 with col3:
#                     button_text = """🔄
#                 Reset
#                 Zoom"""
#                     if st.button(button_text, use_container_width=True):
#                           st.session_state.response_zoom = 1.0
#                           st.session_state.impact_zoom = 1.0
#                           st.rerun()
    
#     # 主界面
#     # 添加Model Information部分
#     st.header("📋 Model Information")
    
#     st.info("""
#     **Model Type**: Ensemble Learning model with six base models (RandomForest, GradientBoosting, 
#     XGBoost, LightGBM, SVR, MLP) 
#     and Stacking ensemble

#     **Optimization Method**: Bayesian hyperparameter optimization

#     **Input**: Structural response time history

#     **Output**: Impact force time history
#     """)

#     st.markdown("---")
    
#     # 修改布局结构，避免嵌套列
#     st.header("📈 Response Data Visualization")
    
#     if st.session_state.data_loaded:
#         dataset = st.session_state.dataset
#         sample_idx = 0  # 现在只有1个样本，选择第0个
        
#         # 创建三列布局：图片、表格、系统状态
#         response_col1, response_col2, response_col3 = st.columns([1, 1.2, 1])
        
#         with response_col1:
#             # 绘制响应曲线 - 调整图片大小与撞击力图片一致
#             response_data = dataset.response_data[sample_idx]
#             time_vector = dataset.time_vector.flatten()
            
#             fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
            
#             # 应用缩放因子
#             zoom_factor = st.session_state.response_zoom
#             if zoom_factor != 1.0:
#                 # 计算缩放后的数据范围
#                 y_min, y_max = response_data.min(), response_data.max()
#                 y_range = y_max - y_min
#                 y_center = (y_min + y_max) / 2
#                 new_y_min = y_center - (y_range * zoom_factor) / 2
#                 new_y_max = y_center + (y_range * zoom_factor) / 2
                
#                 # 设置y轴范围
#                 ax.set_ylim(new_y_min, new_y_max)
            
#             ax.plot(time_vector, response_data, 'b-', linewidth=1.2)
#             ax.set_xlabel('Time (s)', fontsize=10, fontname='Times New Roman')
#             ax.set_ylabel('Displacement (mm)', fontsize=10, fontname='Times New Roman')
#             ax.set_title(f'Response Time History', fontsize=11, fontweight='bold', fontname='Times New Roman')
#             ax.grid(False)  # 关闭网格线
            
#             # 设置刻度字体为Times New Roman
#             for label in ax.get_xticklabels() + ax.get_yticklabels():
#                 label.set_fontname('Times New Roman')
#                 label.set_fontsize(9)
            
#             # 设置图框线
#             for spine in ax.spines.values():
#                 spine.set_linewidth(0.8)
#                 spine.set_color('black')
            
#             plt.tight_layout()
#             st.pyplot(fig)
        
#         with response_col2:
#             # 创建响应数据表格，显示601个点的数据
#             st.subheader("Response Data Table")
            
#             # 创建包含601个点的数据框
#             response_table = pd.DataFrame({
#                 'Time (s)': time_vector,
#                 'Response Data': response_data
#             })
            
#             # 显示完整数据表格
#             st.dataframe(response_table, height=300, use_container_width=True)
            
#             # 添加下载按钮（带图标）
#             csv_response = response_table.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Response Data as CSV",
#                 data=csv_response,
#                 file_name=f"response_data_sample.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )
        
#         with response_col3:
#             st.header("ℹ️ System Status")
            
#             # 使用两个独立的元素显示状态，避免嵌套列
#             if st.session_state.data_loaded:
#                 st.success("✅ Data loaded")
#             else:
#                 st.warning("⚠️ Data not loaded")
            
#             if st.session_state.model_loaded:
#                 st.success("✅ Model loaded")
#             else:
#                 st.warning("⚠️ Model not loaded")
            
#             st.markdown("---")
#             st.subheader("📝 User Guide")
#             st.markdown("""
#             1. **Load Data**: Input data file path and click load
#             2. **Load Model**: Input model file paths and load
#             3. **Make Prediction**: Click to analyze impact force
#             """)
    
#     else:
#         st.info("Please load data first to view response curves")
    
#     # 预测结果展示
#     if st.session_state.prediction_made and st.session_state.data_loaded and st.session_state.model_loaded:
#         st.markdown("---")
#         st.header("📊 Prediction Results Analysis")
        
#         with st.spinner("Making prediction..."):
#             try:
#                 # 获取数据和模型
#                 dataset = st.session_state.dataset
#                 hybrid_model = st.session_state.hybrid_model
                
#                 # 准备数据
#                 X = dataset.features_normalized.numpy()
#                 y_real = dataset.real_normalized.numpy().ravel()
#                 y_imag = dataset.imag_normalized.numpy().ravel()
                
#                 # 由于只有一个样本，我们将整个数据集作为测试集
#                 test_mask = list(range(len(X)))
#                 X_test = X
#                 y_real_test = y_real
#                 y_imag_test = y_imag
                
#                 # 在测试集上进行预测
#                 real_pred_test, imag_pred_test = hybrid_model.predict(X_test)
                
#                 # 计算测试集性能 - 取消显示这部分
#                 real_rmse = np.sqrt(mean_squared_error(y_real_test, real_pred_test))
#                 imag_rmse = np.sqrt(mean_squared_error(y_imag_test, imag_pred_test))
                
#                 # 准备要保存的数据
#                 all_real_pred = real_pred_test
#                 all_imag_pred = imag_pred_test
                
#                 # 可视化唯一的样本
#                 sample_idx = 0
#                 start_idx = sample_idx * dataset.n_freq
#                 end_idx = (sample_idx + 1) * dataset.n_freq
                
#                 # 预测数据（归一化的）
#                 sample_real_pred_norm = all_real_pred[start_idx:end_idx]
#                 sample_imag_pred_norm = all_imag_pred[start_idx:end_idx]
                
#                 # 数据集中的原始实部和虚部数据（未归一化）
#                 sample_real_original = dataset.force_fft_real[sample_idx]
#                 sample_imag_original = dataset.force_fft_imag[sample_idx]
                
#                 # 使用从文件加载的归一化参数反归一化预测数据                   
#                 sample_real_pred_denorm = sample_real_pred_norm * (dataset.real_std + 1e-8) + dataset.real_mean
#                 sample_imag_pred_denorm = sample_imag_pred_norm * (dataset.imag_std + 1e-8) + dataset.imag_mean
                
#                 n_time = dataset.n_time
                
#                 # 构建频域信号
#                 freq_pred_complex = np.zeros(n_time, dtype=complex)
#                 freq_original_complex = np.zeros(n_time, dtype=complex)
                
#                 # 填充正频率部分
#                 freq_pred_complex[:dataset.n_freq] = sample_real_pred_denorm + 1j * sample_imag_pred_denorm
#                 freq_original_complex[:dataset.n_freq] = sample_real_original + 1j * sample_imag_original
                
#                 # 构建共轭对称部分（对于实数时域信号）
#                 # 注意：若在MATLAB中，参数'symmetric'会自动完成此工作
#                 if n_time > 1:
#                     # 对于k=1到n_freq-1，设置负频率部分
#                     # 注意：Python的负频率在数组的后半部分
#                     for k in range(1, min(dataset.n_freq, n_time//2 + 1)):
#                         freq_pred_complex[n_time - k] = np.conj(freq_pred_complex[k])
#                         freq_original_complex[n_time - k] = np.conj(freq_original_complex[k])                    
                
#                 # 执行逆FFT得到归一化的时程
#                 force_pred_time = np.fft.ifft(freq_pred_complex).real
#                 force_original_time = np.fft.ifft(freq_original_complex).real
#                 time_vector = dataset.time_vector.flatten()
                
#                 # 创建三列布局：图片、表格、误差指标
#                 pred_col1, pred_col2, pred_col3 = st.columns([1, 1.2, 1])
                
#                 with pred_col1:
#                     # 只显示第一个图表：撞击力预测
#                     fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
                    
#                     # 应用缩放因子
#                     zoom_factor = st.session_state.impact_zoom
#                     if zoom_factor != 1.0:
#                         # 计算缩放后的数据范围
#                         y_min = min(force_original_time.min(), force_pred_time.min())
#                         y_max = max(force_original_time.max(), force_pred_time.max())
#                         y_range = y_max - y_min
#                         y_center = (y_min + y_max) / 2
#                         new_y_min = y_center - (y_range * zoom_factor) / 2
#                         new_y_max = y_center + (y_range * zoom_factor) / 2
                        
#                         # 设置y轴范围
#                         ax.set_ylim(new_y_min, new_y_max)
                    
#                     # 撞击力时程对比
#                     ax.plot(time_vector, force_original_time, 'b-', 
#                             label='True', linewidth=1.2, alpha=0.7)
#                     ax.plot(time_vector, force_pred_time, 'r--', 
#                             label='Predicted', linewidth=1.2)
#                     ax.set_xlabel('Time (s)', fontname='Times New Roman')
#                     ax.set_ylabel('Impact Force (N)', fontname='Times New Roman')
#                     ax.set_title(f'Impact Force History', fontweight='bold', fontname='Times New Roman')
#                     # 修改图例：边框采用方形矩形，边框线细一点
#                     ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='gray', 
#                               framealpha=1, borderpad=0.3, prop={'family': 'Times New Roman'})
#                     ax.grid(False)  # 关闭网格线
                    
#                     # 设置刻度字体为Times New Roman
#                     for label in ax.get_xticklabels() + ax.get_yticklabels():
#                         label.set_fontname('Times New Roman')
#                         label.set_fontsize(9)
                    
#                     plt.tight_layout()
#                     st.pyplot(fig)
                
#                 with pred_col2:
#                     # 创建数据表格
#                     st.subheader("Prediction Data Table")
                    
#                     # 创建包含601个点的数据框，仅保留前3列，去掉error列
#                     data_table = pd.DataFrame({
#                         'Time (s)': time_vector,
#                         'True Impact Force': force_original_time,
#                         'Predicted Impact Force': force_pred_time
#                     })
                    
#                     # 使用HTML方式显示表格，设置表头字体黑色加粗
#                     st.markdown("""
#                     <style>
#                     .dataframe thead th {
#                         font-weight: bold !important;
#                         color: black !important;
#                     }
#                     </style>
#                     """, unsafe_allow_html=True)
                    
#                     # 显示完整数据表格
#                     st.dataframe(data_table, height=300, use_container_width=True)
                    
#                     # 添加下载按钮（带图标）
#                     csv = data_table.to_csv(index=False)
#                     st.download_button(
#                         label="📥 Download Data as CSV",
#                         data=csv,
#                         file_name=f"prediction_sample_{sample_idx}.csv",
#                         mime="text/csv",
#                         use_container_width=True
#                     )
                
#                 # 修改后的误差指标显示部分
#                 with pred_col3:
#                     # 计算绝对误差指标
#                     mse = np.mean((force_pred_time - force_original_time)**2)
#                     rmse = np.sqrt(mse)
#                     mae = np.mean(np.abs(force_pred_time - force_original_time))
    
#                     # 计算相对误差指标（百分比）
#                     # 使用绝对值的最大值作为分母
#                     max_abs_original = np.max(np.abs(force_original_time))
#                     if max_abs_original > 1e-10:  # 防止除零
#                         rmse_relative = rmse / max_abs_original * 100                        
#                     else:
#                         mse_relative = rmse_relative = mae_relative = max_error_relative = 0
                        
#                     # 计算 R2 指标
#                     # 计算总平方和
#                     ss_total = np.sum((force_original_time - np.mean(force_original_time)) ** 2)
#                     # 计算残差平方和
#                     ss_residual = np.sum((force_original_time - force_pred_time) ** 2)
    
#                     # 避免除零错误
#                     if ss_total > 1e-10:
#                         r2 = 1 - (ss_residual / ss_total)
#                         r2_percent = r2 * 100  # 转换为百分比
#                     else:
#                         # 如果总平方和接近零，说明所有值都相同
#                         # 此时如果预测值也相同且等于真实值，则 R2=1，否则为负无穷
#                         if ss_residual < 1e-10:
#                             r2 = 1.0
#                             r2_percent = 100.0  # 100%
#                         else:
#                             r2 = -float('inf')                                                                
#                             r2_percent = -float('inf')  # 负无穷
                            
#                     st.markdown("### 📊 Error Metrics")
    
#                     # 使用选项卡切换绝对误差和相对误差
#                     tab1, tab2 = st.tabs(["Absolute Errors", "Relative Errors (%)"])
    
#                     with tab1:
#                         st.markdown("**Absolute Error Metrics**")
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.metric("MSE", f"{mse:.4e}")
#                             st.metric("RMSE", f"{rmse:.4e}")
#                         with col2:
#                             st.metric("MAE", f"{mae:.4e}")
        
#                         # 添加说明文本
#                         st.info("""
#                         ℹ️ **Note**: 
#                         - These are absolute error values
#                         - Large values may be due to the large magnitude of impact forces
#                         - See Relative Errors tab for percentage-based metrics
#                         """)
    
#                     with tab2:
#                         st.markdown("**Relative Error Metrics (%)**")
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.metric("NRMSE (%)", f"{rmse_relative:.2f}%")
#                         with col2:
#                             st.metric("R² (%)", f"{r2_percent:.2f}%")
        
#                         # 添加精度评估定性描述
#                         if  rmse_relative < 5:
#                             evaluation = "✅ High Accuracy (Relative RMSE < 5%)"
#                             color = "green"
#                         elif rmse_relative < 10:
#                             evaluation = "👍 Moderate Accuracy (5% ≤ Relative RMSE < 10%)"
#                             color = "blue"
#                         elif rmse_relative < 20:
#                             evaluation = "⚠️ Moderate-Low Accuracy (10% ≤ Relative RMSE < 20%)"
#                             color = "orange"
#                         else:
#                             evaluation = "❌ Low Accuracy (Relative RMSE ≥ 20%)"
#                             color = "red"
        
#                         st.markdown(f"<h4 style='color:{color};'>{evaluation}</h4>", unsafe_allow_html=True)                      
                                       
#             except Exception as e:
#                 st.error(f"Error during prediction: {str(e)}")
#                 import traceback
#                 st.code(traceback.format_exc())
    
#     # 页脚
#     st.markdown("---")
#     st.caption("Vehicle Impact Force Prediction System v1.0 | Based on Ensemble Learning Model")

# if __name__ == "__main__":
#     main()














import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import joblib
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.neural_network import MLPRegressor
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import os  # MODIFIED: 添加 os 和 glob 用于文件查找
import glob

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置页面配置
st.set_page_config(
    page_title="Vehicle Impact Force Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置matplotlib字体 - 改回Times New Roman
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'figure.frameon': True,
    'figure.dpi': 150,
})


class HybridBridgeImpactDataset(Dataset):
    """混合模型数据集"""
    def __init__(self, mat_path, normalize=True):
        # 加载数据
        data = sio.loadmat(mat_path)
        
        # 获取数据 
        # 注意：新数据集vehicle_and_bridge_collision_dataset.mat只有1条样本
        self.force_data = data['force_data']  # (1, 601)
        self.response_data = data['response_data']  # (1, 601)
        self.force_fft_real = data['force_fft_real']  # (1, 150)
        self.force_fft_imag = data['force_fft_imag']  # (1, 150)
        self.freq_vector = data['freq_vector']  # (1, 150)
        self.time_vector = data['time_vector']  # (1, 601)
        self.sampling_rate = float(data['sampling_rate'])
        
        # 转换为numpy数组（与训练代码一致）
        self.force_data = np.array(self.force_data)
        self.response_data = np.array(self.response_data)
        self.force_fft_real = np.array(self.force_fft_real)
        self.force_fft_imag = np.array(self.force_fft_imag)
        self.freq_vector = np.array(self.freq_vector)
        self.time_vector = np.array(self.time_vector)

        # MODIFIED: 确保关键数组至少是2维（样本数×特征数）
        if self.force_data.ndim == 0:
            self.force_data = np.array([[self.force_data.item()]])
        elif self.force_data.ndim == 1:
            self.force_data = self.force_data.reshape(1, -1)

        if self.response_data.ndim == 0:
            self.response_data = np.array([[self.response_data.item()]])
        elif self.response_data.ndim == 1:
            self.response_data = self.response_data.reshape(1, -1)

        if self.force_fft_real.ndim == 0:
            self.force_fft_real = np.array([[self.force_fft_real.item()]])
        elif self.force_fft_real.ndim == 1:
            self.force_fft_real = self.force_fft_real.reshape(1, -1)

        if self.force_fft_imag.ndim == 0:
            self.force_fft_imag = np.array([[self.force_fft_imag.item()]])
        elif self.force_fft_imag.ndim == 1:
            self.force_fft_imag = self.force_fft_imag.reshape(1, -1)

        if self.freq_vector.ndim == 0:
            self.freq_vector = np.array([self.freq_vector.item()])
        elif self.freq_vector.ndim > 1:
            self.freq_vector = self.freq_vector.flatten()

        if self.time_vector.ndim == 0:
            self.time_vector = np.array([self.time_vector.item()])
        elif self.time_vector.ndim > 1:
            self.time_vector = self.time_vector.flatten()
        
        # 数据形状
        self.n_samples = self.force_data.shape[0]  # 现在为1
        self.n_time = self.force_data.shape[1]  # 601
        self.n_freq = self.force_fft_real.shape[1]  # 150
        
        # 提取时域和频域特征
        self.extract_features()
        
        # 准备训练数据
        self.prepare_expanded_dataset(normalize)
    
    def extract_features(self):
        """提取时域和频域特征"""
        # 时域特征
        time_features = []
        for i in range(self.n_samples):
            response = self.response_data[i]
            
            # 基本统计特征
            mean_val = response.mean()
            std_val = response.std()
            min_val = response.min()
            
            # 峰值特征
            max_val = response.max()
            rms_val = np.sqrt(np.mean(response**2))
            peak_to_peak = max_val - min_val
            crest_factor = max_val / rms_val if rms_val != 0 else 0
            shape_factor = rms_val / np.mean(np.abs(response)) if np.mean(np.abs(response)) != 0 else 0
            
            time_features.append([
                mean_val, std_val, min_val,  # 保留的3个基本特征
                peak_to_peak, crest_factor, shape_factor  # 保留的3个峰值特征
            ])
        
        self.time_features = np.array(time_features)  # (n_samples, 6)
        
        # 频域特征
        freq_features = []
        for i in range(self.n_samples):
            response = self.response_data[i]
            fft_result = np.fft.fft(response)[:self.n_freq]  # 取前n_freq个频率
            
            # 频域统计特征
            mag_spectrum = np.abs(fft_result)
            
            # 基本频域统计
            freq_mean = mag_spectrum.mean()
            freq_std = mag_spectrum.std()
            freq_max = mag_spectrum.max()
            freq_min = mag_spectrum.min()
            
            # 频域形状特征
            freq_skewness = self._skewness(mag_spectrum)
            freq_kurtosis = self._kurtosis(mag_spectrum)
            
            sample_features = [
                freq_mean, freq_std, freq_min, freq_max,  # 保留的4个基本统计特征
                freq_skewness, freq_kurtosis  # 保留的2个形状特征
            ]
            
            freq_features.append(sample_features)
        
        self.freq_features = np.array(freq_features)  # (n_samples, 6)
    
    def _skewness(self, x):
        """计算偏度"""
        n = len(x)
        if n < 2:
            return 0
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        skew = np.mean((x - mean)**3) / (std**3)
        return skew
    
    def _kurtosis(self, x):
        """计算峰度"""
        n = len(x)
        if n < 2:
            return 0
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        kurt = np.mean((x - mean)**4) / (std**4) - 3  # 超额峰度
        return kurt
    
    def prepare_expanded_dataset(self, normalize=True):
        """准备扩展的数据集 """
        # 扩展响应数据：每个样本重复n_freq次
        response_expanded = np.repeat(self.response_data, self.n_freq, axis=0)  # (n_samples*n_freq, 601)
        
        # 扩展时域特征
        time_features_expanded = np.repeat(self.time_features, self.n_freq, axis=0)  # (n_samples*n_freq, 6)
        
        # 扩展频域特征
        freq_features_expanded = np.repeat(self.freq_features, self.n_freq, axis=0)  # (n_samples*n_freq, 6)
        
        # 扩展频率：为每个样本的每个频率创建频率向量
        freq_expanded = np.tile(self.freq_vector.T, (self.n_samples, 1)).reshape(-1, 1)  # (n_samples*n_freq, 1)
        
        # 扩展FFT实部和虚部
        real_expanded = self.force_fft_real.reshape(-1, 1)  # (n_samples*n_freq, 1)
        imag_expanded = self.force_fft_imag.reshape(-1, 1)  # (n_samples*n_freq, 1)
        
        # 组合所有特征
        combined_features = np.hstack([
            response_expanded,  # 原始时程数据 (601)
            time_features_expanded,  # 时域特征 (6)
            freq_features_expanded,  # 频域特征 (6)
            freq_expanded  # 频率 (1)
        ])  # 总共 614 维特征
        
        # 归一化
        if normalize:
            # 归一化特征            
            # 加载归一化参数文件中的均值和标准差
            norm_params = sio.loadmat('normalization_parameters.mat')
            self.feature_mean = norm_params['feature_mean']
            self.feature_std = norm_params['feature_std']           
            self.real_mean = norm_params['real_mean'].item()
            self.real_std = norm_params['real_std'].item()
            self.imag_mean = norm_params['imag_mean'].item()
            self.imag_std = norm_params['imag_std'].item()
            
            # 使用从文件加载的参数进行归一化
            self.features_normalized = (combined_features - self.feature_mean) / (self.feature_std + 1e-8)
            self.real_normalized = (real_expanded - self.real_mean) / (self.real_std + 1e-8)
            self.imag_normalized = (imag_expanded - self.imag_mean) / (self.imag_std + 1e-8)
        else:
            self.features_normalized = combined_features
            self.real_normalized = real_expanded
            self.imag_normalized = imag_expanded
        
        # 存储原始数据用于重建
        self.response_expanded = response_expanded
        self.freq_expanded = freq_expanded
        self.real_expanded = real_expanded
        self.imag_expanded = imag_expanded
        
        self.expanded_size = len(self.features_normalized)
        
        # 转换为torch张量以保持与原始代码兼容
        self.features_normalized = torch.FloatTensor(self.features_normalized)
        self.real_normalized = torch.FloatTensor(self.real_normalized)
        self.imag_normalized = torch.FloatTensor(self.imag_normalized)
        self.response_expanded = torch.FloatTensor(self.response_expanded)
        self.freq_expanded = torch.FloatTensor(self.freq_expanded)
        self.real_expanded = torch.FloatTensor(self.real_expanded)
        self.imag_expanded = torch.FloatTensor(self.imag_expanded)
    
    def __len__(self):
        return self.expanded_size
    
    def __getitem__(self, idx):
        return {
            'features': self.features_normalized[idx],
            'real': self.real_normalized[idx],
            'imag': self.imag_normalized[idx],
            'response_original': self.response_expanded[idx],
            'freq_original': self.freq_expanded[idx],
            'real_original': self.real_expanded[idx],
            'imag_original': self.imag_expanded[idx]
        }

class CustomMLPRegressor(MLPRegressor):
    """自定义MLP回归器，用于处理字符串表示的隐藏层结构"""
    def set_params(self, **params):
        # 转换字符串表示的隐藏层结构
        if 'hidden_layer_sizes_str' in params:
            hidden_str = params.pop('hidden_layer_sizes_str')
            # 将字符串转换为元组
            if '_' in hidden_str:
                params['hidden_layer_sizes'] = tuple(int(x) for x in hidden_str.split('_'))
            else:
                params['hidden_layer_sizes'] = int(hidden_str)
        return super().set_params(**params)

class HybridModel:
    """机器学习模型"""
    def __init__(self):
        self.ml_models = {'real': [], 'imag': []}
        self.meta_models = {'real': None, 'imag': None}
        
    def load_models(self, ml_model_path, meta_model_path):
        """加载已训练的模型"""
        # 加载机器学习模型
        ml_models_data = joblib.load(ml_model_path)
        
        # 重新构建ml_models结构
        for name, model_list in ml_models_data['real']:
            self.ml_models['real'].append((name, model_list))
        
        for name, model_list in ml_models_data['imag']:
            self.ml_models['imag'].append((name, model_list))
        
        # 加载元模型
        self.meta_models = joblib.load(meta_model_path)
        
        return self
    
    def predict(self, X):
        """使用混合模型进行预测"""
        # 获取基模型预测
        real_base_preds = []
        imag_base_preds = []
        
        # 机器学习模型预测
        for name, model_list in self.ml_models['real']:
            model = model_list[0]  # 获取单个模型
            predictions = model.predict(X)
            real_base_preds.append(predictions.reshape(-1, 1))
        
        for name, model_list in self.ml_models['imag']:
            model = model_list[0]  # 获取单个模型
            predictions = model.predict(X)
            imag_base_preds.append(predictions.reshape(-1, 1))
        
        # 合并所有预测
        real_features = np.hstack(real_base_preds)
        imag_features = np.hstack(imag_base_preds)
        
        # 元模型预测
        real_pred_final = self.meta_models['real'].predict(real_features)
        imag_pred_final = self.meta_models['imag'].predict(imag_features)
        
        return real_pred_final, imag_pred_final

# 初始化session state中的缩放因子
if 'response_zoom' not in st.session_state:
    st.session_state.response_zoom = 1.0
if 'impact_zoom' not in st.session_state:
    st.session_state.impact_zoom = 1.0

# MODIFIED: 添加文件查找辅助函数
def find_mat_file():
    """查找当前目录下的.mat文件，返回第一个找到的文件名"""
    mat_files = glob.glob("*.mat")
    if mat_files:
        return mat_files[1]
    return None

# 主应用程序
def main():
    # 创建标题栏布局，将标题和校徽放在同一行
    col_title, col_logo = st.columns([3, 1])
    
    with col_title:
        # 标题左对齐，设置深蓝色（RGB: 0, 51, 102）
        st.markdown("<h1 style='text-align: left; color: rgb(0, 51, 102); margin-bottom: 3px;'>🚗 Vehicle Impact Force Prediction System for Bridge Piers</h1>", unsafe_allow_html=True)
    
    with col_logo:
        # 校徽右对齐
        try:
            # 创建一个右对齐的容器
            st.markdown(
                """
                <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image("seu1_logo.png", width=250)
        except:
            st.markdown("<div style='display: flex; justify-content: flex-end;'><strong>SEU Logo</strong></div>", unsafe_allow_html=True)
    
    # 添加与标题颜色一致的横线，加粗并靠近标题
    st.markdown("<hr style='height:4px; border-width:0; color:rgb(0, 51, 102); background-color:rgb(0, 51, 102); margin-top: 0; margin-bottom: 18px;'>", unsafe_allow_html=True)
    
    # 开发者信息 - 左对齐（现在放在第二行）
    st.markdown("<h4 style='text-align: left; margin-bottom: 0;'>Developed by B.Q. Wang, D. Feng, et al.</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left; margin-top: 0;'>Southeast University (SEU), Nanjing, Jiangsu, China</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;'>E-mail: bqwang@seu.edu.cn</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 初始化session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'hybrid_model' not in st.session_state:
        st.session_state.hybrid_model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 Data Control")
        
        # 数据加载部分
        st.subheader("1. Load Data")
        # MODIFIED: 自动检测数据文件
        default_file = find_mat_file()
        if default_file:
            st.success(f"Detected data file: {default_file}")
            data_path = st.text_input("Data file path (can edit)", default_file)
        else:
            st.warning("No .mat file detected, please enter path manually")
            data_path = st.text_input("Data file path", "vehicle_and_bridge_collision_dataset.mat")
        
        if st.button("📂 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                try:
                    # MODIFIED: 检查文件是否存在
                    if not os.path.exists(data_path):
                        st.error(f"File not found: {data_path}")
                        st.info("Current directory contents: " + ", ".join(os.listdir('.')))
                        st.stop()
                    
                    dataset = HybridBridgeImpactDataset(data_path, normalize=True)
                    st.session_state.dataset = dataset
                    st.session_state.data_loaded = True
                    st.success(f"Data loaded successfully! {dataset.n_samples} sample in total")
                except Exception as e:
                    st.error(f"Failed to load data: {str(e)}")
                    # MODIFIED: 显示更多调试信息
                    import traceback
                    st.code(traceback.format_exc())
        
        if st.session_state.data_loaded:
            st.info(f"✅ Data loaded")
            
            # 模型加载部分
            st.subheader("2. Load Model")
            ml_model_path = st.text_input("Base model path", "hybrid_ml_models.joblib")
            meta_model_path = st.text_input("Meta model path", "hybrid_meta_models.joblib")
            
            if st.button("🤖 Load Model", use_container_width=True):
                with st.spinner("Loading model..."):
                    try:
                        hybrid_model = HybridModel()
                        hybrid_model.load_models(ml_model_path, meta_model_path)
                        st.session_state.hybrid_model = hybrid_model
                        st.session_state.model_loaded = True
                        st.success("Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
        
        if st.session_state.model_loaded:
            st.info("✅ Model loaded")
            
            # 预测按钮
            if st.session_state.data_loaded:
                st.subheader("3. Make Prediction")
                if st.button("🔮 Predict Impact Force", use_container_width=True, type="primary"):
                    st.session_state.prediction_made = True
                    st.rerun()
                
                # 添加Plot Controls组件
                st.subheader("🔍 Plot Controls")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # 使用不可见字符强制三行布局
                    button_text = """➕
                Zoom
                in"""
                    if st.button(button_text, use_container_width=True):
                        st.session_state.response_zoom *= 0.8
                        st.session_state.impact_zoom *= 0.8
                        st.rerun()

                with col2:
                    button_text = """➖
                Zoom
                out"""
                    if st.button(button_text, use_container_width=True):
                          st.session_state.response_zoom *= 1.25
                          st.session_state.impact_zoom *= 1.25
                          st.rerun()

                with col3:
                    button_text = """🔄
                Reset
                Zoom"""
                    if st.button(button_text, use_container_width=True):
                          st.session_state.response_zoom = 1.0
                          st.session_state.impact_zoom = 1.0
                          st.rerun()
    
    # 主界面
    # 添加Model Information部分
    st.header("📋 Model Information")
    
    st.info("""
    **Model Type**: Ensemble Learning model with six base models (RandomForest, GradientBoosting, 
    XGBoost, LightGBM, SVR, MLP) 
    and Stacking ensemble

    **Optimization Method**: Bayesian hyperparameter optimization

    **Input**: Structural response time history

    **Output**: Impact force time history
    """)

    st.markdown("---")
    
    # 修改布局结构，避免嵌套列
    st.header("📈 Response Data Visualization")
    
    if st.session_state.data_loaded:
        dataset = st.session_state.dataset
        sample_idx = 0  # 现在只有1个样本，选择第0个
        
        # 创建三列布局：图片、表格、系统状态
        response_col1, response_col2, response_col3 = st.columns([1, 1.2, 1])
        
        with response_col1:
            # 绘制响应曲线 - 调整图片大小与撞击力图片一致
            response_data = dataset.response_data[sample_idx]
            time_vector = dataset.time_vector.flatten()
            
            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
            
            # 应用缩放因子
            zoom_factor = st.session_state.response_zoom
            if zoom_factor != 1.0:
                # 计算缩放后的数据范围
                y_min, y_max = response_data.min(), response_data.max()
                y_range = y_max - y_min
                y_center = (y_min + y_max) / 2
                new_y_min = y_center - (y_range * zoom_factor) / 2
                new_y_max = y_center + (y_range * zoom_factor) / 2
                
                # 设置y轴范围
                ax.set_ylim(new_y_min, new_y_max)
            
            ax.plot(time_vector, response_data, 'b-', linewidth=1.2)
            ax.set_xlabel('Time (s)', fontsize=10, fontname='Times New Roman')
            ax.set_ylabel('Displacement (mm)', fontsize=10, fontname='Times New Roman')
            ax.set_title(f'Response Time History', fontsize=11, fontweight='bold', fontname='Times New Roman')
            ax.grid(False)  # 关闭网格线
            
            # 设置刻度字体为Times New Roman
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(9)
            
            # 设置图框线
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('black')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with response_col2:
            # 创建响应数据表格，显示601个点的数据
            st.subheader("Response Data Table")
            
            # 创建包含601个点的数据框
            response_table = pd.DataFrame({
                'Time (s)': time_vector,
                'Response Data': response_data
            })
            
            # 显示完整数据表格
            st.dataframe(response_table, height=300, use_container_width=True)
            
            # 添加下载按钮（带图标）
            csv_response = response_table.to_csv(index=False)
            st.download_button(
                label="📥 Download Response Data as CSV",
                data=csv_response,
                file_name=f"response_data_sample.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with response_col3:
            st.header("ℹ️ System Status")
            
            # 使用两个独立的元素显示状态，避免嵌套列
            if st.session_state.data_loaded:
                st.success("✅ Data loaded")
            else:
                st.warning("⚠️ Data not loaded")
            
            if st.session_state.model_loaded:
                st.success("✅ Model loaded")
            else:
                st.warning("⚠️ Model not loaded")
            
            st.markdown("---")
            st.subheader("📝 User Guide")
            st.markdown("""
            1. **Load Data**: Input data file path and click load
            2. **Load Model**: Input model file paths and load
            3. **Make Prediction**: Click to analyze impact force
            """)
    
    else:
        st.info("Please load data first to view response curves")
    
    # 预测结果展示
    if st.session_state.prediction_made and st.session_state.data_loaded and st.session_state.model_loaded:
        st.markdown("---")
        st.header("📊 Prediction Results Analysis")
        
        with st.spinner("Making prediction..."):
            try:
                # 获取数据和模型
                dataset = st.session_state.dataset
                hybrid_model = st.session_state.hybrid_model
                
                # 准备数据
                X = dataset.features_normalized.numpy()
                y_real = dataset.real_normalized.numpy().ravel()
                y_imag = dataset.imag_normalized.numpy().ravel()
                
                # 由于只有一个样本，我们将整个数据集作为测试集
                test_mask = list(range(len(X)))
                X_test = X
                y_real_test = y_real
                y_imag_test = y_imag
                
                # 在测试集上进行预测
                real_pred_test, imag_pred_test = hybrid_model.predict(X_test)
                
                # 计算测试集性能 - 取消显示这部分
                real_rmse = np.sqrt(mean_squared_error(y_real_test, real_pred_test))
                imag_rmse = np.sqrt(mean_squared_error(y_imag_test, imag_pred_test))
                
                # 准备要保存的数据
                all_real_pred = real_pred_test
                all_imag_pred = imag_pred_test
                
                # 可视化唯一的样本
                sample_idx = 0
                start_idx = sample_idx * dataset.n_freq
                end_idx = (sample_idx + 1) * dataset.n_freq
                
                # 预测数据（归一化的）
                sample_real_pred_norm = all_real_pred[start_idx:end_idx]
                sample_imag_pred_norm = all_imag_pred[start_idx:end_idx]
                
                # 数据集中的原始实部和虚部数据（未归一化）
                sample_real_original = dataset.force_fft_real[sample_idx]
                sample_imag_original = dataset.force_fft_imag[sample_idx]
                
                # 使用从文件加载的归一化参数反归一化预测数据                   
                sample_real_pred_denorm = sample_real_pred_norm * (dataset.real_std + 1e-8) + dataset.real_mean
                sample_imag_pred_denorm = sample_imag_pred_norm * (dataset.imag_std + 1e-8) + dataset.imag_mean
                
                n_time = dataset.n_time
                
                # 构建频域信号
                freq_pred_complex = np.zeros(n_time, dtype=complex)
                freq_original_complex = np.zeros(n_time, dtype=complex)
                
                # 填充正频率部分
                freq_pred_complex[:dataset.n_freq] = sample_real_pred_denorm + 1j * sample_imag_pred_denorm
                freq_original_complex[:dataset.n_freq] = sample_real_original + 1j * sample_imag_original
                
                # 构建共轭对称部分（对于实数时域信号）
                # 注意：若在MATLAB中，参数'symmetric'会自动完成此工作
                if n_time > 1:
                    # 对于k=1到n_freq-1，设置负频率部分
                    # 注意：Python的负频率在数组的后半部分
                    for k in range(1, min(dataset.n_freq, n_time//2 + 1)):
                        freq_pred_complex[n_time - k] = np.conj(freq_pred_complex[k])
                        freq_original_complex[n_time - k] = np.conj(freq_original_complex[k])                    
                
                # 执行逆FFT得到归一化的时程
                force_pred_time = np.fft.ifft(freq_pred_complex).real
                force_original_time = np.fft.ifft(freq_original_complex).real
                time_vector = dataset.time_vector.flatten()
                
                # 创建三列布局：图片、表格、误差指标
                pred_col1, pred_col2, pred_col3 = st.columns([1, 1.2, 1])
                
                with pred_col1:
                    # 只显示第一个图表：撞击力预测
                    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
                    
                    # 应用缩放因子
                    zoom_factor = st.session_state.impact_zoom
                    if zoom_factor != 1.0:
                        # 计算缩放后的数据范围
                        y_min = min(force_original_time.min(), force_pred_time.min())
                        y_max = max(force_original_time.max(), force_pred_time.max())
                        y_range = y_max - y_min
                        y_center = (y_min + y_max) / 2
                        new_y_min = y_center - (y_range * zoom_factor) / 2
                        new_y_max = y_center + (y_range * zoom_factor) / 2
                        
                        # 设置y轴范围
                        ax.set_ylim(new_y_min, new_y_max)
                    
                    # 撞击力时程对比
                    ax.plot(time_vector, force_original_time, 'b-', 
                            label='True', linewidth=1.2, alpha=0.7)
                    ax.plot(time_vector, force_pred_time, 'r--', 
                            label='Predicted', linewidth=1.2)
                    ax.set_xlabel('Time (s)', fontname='Times New Roman')
                    ax.set_ylabel('Impact Force (N)', fontname='Times New Roman')
                    ax.set_title(f'Impact Force History', fontweight='bold', fontname='Times New Roman')
                    # 修改图例：边框采用方形矩形，边框线细一点
                    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='gray', 
                              framealpha=1, borderpad=0.3, prop={'family': 'Times New Roman'})
                    ax.grid(False)  # 关闭网格线
                    
                    # 设置刻度字体为Times New Roman
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontname('Times New Roman')
                        label.set_fontsize(9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with pred_col2:
                    # 创建数据表格
                    st.subheader("Prediction Data Table")
                    
                    # 创建包含601个点的数据框，仅保留前3列，去掉error列
                    data_table = pd.DataFrame({
                        'Time (s)': time_vector,
                        'True Impact Force': force_original_time,
                        'Predicted Impact Force': force_pred_time
                    })
                    
                    # 使用HTML方式显示表格，设置表头字体黑色加粗
                    st.markdown("""
                    <style>
                    .dataframe thead th {
                        font-weight: bold !important;
                        color: black !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 显示完整数据表格
                    st.dataframe(data_table, height=300, use_container_width=True)
                    
                    # 添加下载按钮（带图标）
                    csv = data_table.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"prediction_sample_{sample_idx}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # 修改后的误差指标显示部分
                with pred_col3:
                    # 计算绝对误差指标
                    mse = np.mean((force_pred_time - force_original_time)**2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(force_pred_time - force_original_time))
    
                    # 计算相对误差指标（百分比）
                    # 使用绝对值的最大值作为分母
                    max_abs_original = np.max(np.abs(force_original_time))
                    if max_abs_original > 1e-10:  # 防止除零
                        rmse_relative = rmse / max_abs_original * 100                        
                    else:
                        mse_relative = rmse_relative = mae_relative = max_error_relative = 0
                        
                    # 计算 R2 指标
                    # 计算总平方和
                    ss_total = np.sum((force_original_time - np.mean(force_original_time)) ** 2)
                    # 计算残差平方和
                    ss_residual = np.sum((force_original_time - force_pred_time) ** 2)
    
                    # 避免除零错误
                    if ss_total > 1e-10:
                        r2 = 1 - (ss_residual / ss_total)
                        r2_percent = r2 * 100  # 转换为百分比
                    else:
                        # 如果总平方和接近零，说明所有值都相同
                        # 此时如果预测值也相同且等于真实值，则 R2=1，否则为负无穷
                        if ss_residual < 1e-10:
                            r2 = 1.0
                            r2_percent = 100.0  # 100%
                        else:
                            r2 = -float('inf')                                                                
                            r2_percent = -float('inf')  # 负无穷
                            
                    st.markdown("### 📊 Error Metrics")
    
                    # 使用选项卡切换绝对误差和相对误差
                    tab1, tab2 = st.tabs(["Absolute Errors", "Relative Errors (%)"])
    
                    with tab1:
                        st.markdown("**Absolute Error Metrics**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", f"{mse:.4e}")
                            st.metric("RMSE", f"{rmse:.4e}")
                        with col2:
                            st.metric("MAE", f"{mae:.4e}")
        
                        # 添加说明文本
                        st.info("""
                        ℹ️ **Note**: 
                        - These are absolute error values
                        - Large values may be due to the large magnitude of impact forces
                        - See Relative Errors tab for percentage-based metrics
                        """)
    
                    with tab2:
                        st.markdown("**Relative Error Metrics (%)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("NRMSE (%)", f"{rmse_relative:.2f}%")
                        with col2:
                            st.metric("R² (%)", f"{r2_percent:.2f}%")
        
                        # 添加精度评估定性描述
                        if  rmse_relative < 5:
                            evaluation = "✅ High Accuracy (Relative RMSE < 5%)"
                            color = "green"
                        elif rmse_relative < 10:
                            evaluation = "👍 Moderate Accuracy (5% ≤ Relative RMSE < 10%)"
                            color = "blue"
                        elif rmse_relative < 20:
                            evaluation = "⚠️ Moderate-Low Accuracy (10% ≤ Relative RMSE < 20%)"
                            color = "orange"
                        else:
                            evaluation = "❌ Low Accuracy (Relative RMSE ≥ 20%)"
                            color = "red"
        
                        st.markdown(f"<h4 style='color:{color};'>{evaluation}</h4>", unsafe_allow_html=True)                      
                                       
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # 页脚
    st.markdown("---")
    st.caption("Vehicle Impact Force Prediction System v1.0 | Based on Ensemble Learning Model")

if __name__ == "__main__":
    main()






