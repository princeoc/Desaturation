import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 页面基础设置 ---
st.set_page_config(page_title="临床风险预测模型", layout="centered")

# --- 加载训练好的模型和相关信息 ---
@st.cache_resource
def load_model_data():
    try:
        saved_object = joblib.load('lasso_model.joblib')
        return saved_object
    except FileNotFoundError:
        return None

saved_data = load_model_data()
model = None
risk_threshold = 0.5  # 默认阈值，如果模型文件中没有，则使用此值
feature_names = []

if saved_data and isinstance(saved_data, dict):
    model = saved_data.get('model')
    risk_threshold = saved_data.get('threshold', 0.5)
    feature_names = saved_data.get('features')

# --- 主页面 ---
st.title('术中低氧血症风险预测工具')

st.markdown("---")

# --- 输入区域 ---
if model:
    st.header("患者信息输入")

    # 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('年龄 (岁)', 18, 100, 50)
        # **改动1: 接收原始身高体重输入**
        height_cm = st.number_input('身高 (cm)', min_value=100.0, max_value=220.0, value=170.0, step=0.5)
        weight_kg = st.number_input('体重 (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        
    with col2:
        # **改动2: 接收原始颈围输入**
        neck_circumference_cm = st.number_input('颈围 (cm)', min_value=20.0, max_value=60.0, value=38.0, step=0.5)
        asa_grade = st.selectbox('ASA分级', [1, 2, 3, 4], index=0)
        base_spo2 = st.slider('基础脉氧 (%)', 80, 100, 98)

    # 自动计算并显示衍征指标
    st.subheader("自动计算指标")
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        if height_cm > 0:
            bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
            st.metric(label="BMI值", value=f"{bmi_calculated:.2f}")
        else:
            bmi_calculated = 0
            st.warning("身高不能为0")
            
    with col_calc2:
        if height_cm > 0:
            # **改动3: 修正颈高比计算公式，与训练数据保持一致**
            neck_height_ratio_calculated = neck_circumference_cm / height_cm 
            st.metric(label="颈高比", value=f"{neck_height_ratio_calculated:.3f}")
        else:
            neck_height_ratio_calculated = 0

    # --- 按钮和输出区域 ---
    st.markdown("---")
    if st.button('计算风险等级', use_container_width=True):
        
        # **改动4: 使用后台计算出的指标构建输入数据**
        input_data = {
            'age': age,
            'bmi': bmi_calculated,
            'neck_height_ratio': neck_height_ratio_calculated,
            'asa_grade': asa_grade,
            'base_spo2': base_spo2
        }
        
        # 确保输入DataFrame的列顺序与训练时完全一致
        input_df = pd.DataFrame([input_data])[feature_names]

        # --- 预测与展示 ---
        prediction_proba = model.predict_proba(input_df)[0][1]
        risk_level = "高风险" if prediction_proba >= risk_threshold else "低风险"
        
        st.header('预测结果')
        
        if risk_level == "高风险":
            st.error(f"风险等级: {risk_level}", icon="⚠️")
        else:
            st.success(f"风险等级: {risk_level}", icon="✅")

else:
    st.error("错误：找不到模型文件 `lasso_model.joblib`。请先运行 `train_lasso_model.py` 脚本来生成模型文件。")

