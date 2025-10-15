# app.py

import streamlit as st
import pandas as pd
import joblib

# --- 页面基础设置 ---
st.set_page_config(page_title="临床风险预测工具", layout="centered")

# --- 加载训练好的模型 ---
# @st.cache_resource 确保模型只加载一次，提高效率
@st.cache_resource
def load_model():
    # 从文件加载我们之前训练好的整个管道
    try:
        return joblib.load('final_lgbm_model.joblib')
    except FileNotFoundError:
        return None

model = load_model()

# --- 主页面 ---
st.title('术中低氧血症风险预测工具')

if model is None:
    st.error("错误：找不到模型文件 'final_lgbm_model.joblib'。请先运行 'train_model.py' 脚本来生成模型文件。")
else:
    st.write("请在下方输入患者的临床指标，然后点击“计算风险”按钮。")
    
    # 使用列布局创建输入界面
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('年龄 (岁)', min_value=0, max_value=120, value=50)
        sex = st.selectbox('性别', ['女', '男'], index=1)
        asa_grade = st.selectbox('ASA分级', [1, 2, 3, 4], index=1)

    with col2:
        bmi = st.number_input('体重指数 (BMI)', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        neck_circumference = st.number_input('颈围 (cm)', min_value=20.0, max_value=70.0, value=40.0, step=0.5)
        mallampati_grade = st.selectbox('马氏分级', [1, 2, 3, 4], index=1)
        
    with col3:
        hypertension = st.selectbox('高血压', ['无', '有'], index=0)
        diabetes = st.selectbox('糖尿病', ['无', '有'], index=0)
        osahs = st.selectbox('OSAHS', ['无', '有'], index=0)

    # --- 预测逻辑 ---
    if st.button('计算风险', type="primary"):
        
        # 准备模型所需的完整特征列表（与训练时完全一致）
        feature_names = [
            'age', 'sex', 'height_cm', 'weight_kg', 'bmi', 'neck_circumference_cm', 'neck_height_ratio', 'neck_length_cm',
            'asa_grade', 'mallampati_grade', 'hypertension', 'diabetes', 'heart_disease', 'osahs', 'smoking',
            'alcoholism', 'snoring', 'sleeping_pills', 'base_systolic_bp', 'base_diastolic_bp', 'base_map',
            'base_hr', 'base_spo2'
        ]

        # 将用户的输入转换为模型需要的数值格式
        sex_value = 1 if sex == '男' else 0
        hypertension_value = 1 if hypertension == '有' else 0
        diabetes_value = 1 if diabetes == '有' else 0
        osahs_value = 1 if osahs == '有' else 0

        # 创建一个字典来收集所有输入
        # 【重要】对于没有让用户输入的特征，我们暂时使用平均值或常见值作为默认值
        # 这样可以确保模型能运行。在实际应用中，您应该添加更多重要的输入项。
        input_data = {
            'age': age, 'sex': sex_value, 'bmi': bmi, 'neck_circumference_cm': neck_circumference,
            'asa_grade': asa_grade, 'mallampati_grade': mallampati_grade, 
            'hypertension': hypertension_value, 'diabetes': diabetes_value, 'osahs': osahs_value,
            # --- 使用默认值填充其余特征 ---
            'height_cm': 170, 'weight_kg': 70, 'neck_height_ratio': 22, 'neck_length_cm': 14, 
            'heart_disease': 0, 'smoking': 0, 'alcoholism': 0, 'snoring': 0, 'sleeping_pills': 0, 
            'base_systolic_bp': 120, 'base_diastolic_bp': 80, 'base_map': 93, 
            'base_hr': 75, 'base_spo2': 98
        }
        
        # 转换为DataFrame，并确保列顺序正确
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # 使用加载的模型进行预测
        prediction_prob = model.predict_proba(input_df)[0][1] # 获取“有风险”(类别为1)的概率
        
        st.write("---")
        st.subheader('预测结果')
        
        # 使用大号字体和颜色来突出显示结果
        if prediction_prob >= 0.5:
            st.markdown(f'#### <p style="color:red;">该患者的风险概率为: {prediction_prob:.1%}</p>', unsafe_allow_html=True)
            st.warning('**建议关注**：预测风险较高。')
        else:
            st.markdown(f'#### <p style="color:green;">该患者的风险概率为: {prediction_prob:.1%}</p>', unsafe_allow_html=True)
            st.success('**风险较低**。')
