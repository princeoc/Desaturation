import streamlit as st
import pandas as pd
import joblib

# --- 页面基础设置 ---
st.set_page_config(page_title="临床风险预测模型", layout="wide")

# --- 加载训练好的模型 ---
@st.cache_resource
def load_model():
    # 注意：文件名已更新为 'final_model.joblib'
    try:
        return joblib.load('final_model.joblib')
    except FileNotFoundError:
        return None

model = load_model()

# --- 侧边栏：用户输入 ---
st.sidebar.header('请输入患者信息')

# 创建身高和体重的输入框，用于自动计算BMI
height_cm = st.sidebar.number_input('身高 (cm)', min_value=100.0, max_value=220.0, value=170.0, step=0.5)
weight_kg = st.sidebar.number_input('体重 (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.5)

# 自动计算BMI并显示
if height_cm > 0:
    bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
    st.sidebar.metric(label="计算出的BMI值", value=f"{bmi_calculated:.2f}")
else:
    bmi_calculated = 0
    st.sidebar.error("身高不能为0")

# 其他特征的输入
age = st.sidebar.slider('年龄 (岁)', 18, 100, 50)
neck_circumference = st.sidebar.slider('颈围 (cm)', 20, 60, 38)
asa_grade = st.sidebar.selectbox('ASA分级', [1, 2, 3, 4], index=0)
hypertension = st.sidebar.selectbox('是否有高血压', ['无', '有'], index=0)
snoring = st.sidebar.selectbox('是否打鼾', ['否', '是'], index=0)

# 模型训练时使用的所有特征名称列表
feature_names = [
    'age', 'sex', 'height_cm', 'weight_kg', 'bmi', 'neck_circumference_cm', 'neck_height_ratio', 'neck_length_cm',
    'asa_grade', 'mallampati_grade', 'hypertension', 'diabetes', 'heart_disease', 'osahs', 'smoking',
    'alcoholism', 'snoring', 'sleeping_pills', 'base_systolic_bp', 'base_diastolic_bp', 'base_map',
    'base_hr', 'base_spo2'
]

# --- 主页面 ---
st.title('术中低氧血症风险预测工具')
st.markdown("---")

# 检查模型是否已加载
if model is None:
    st.error("错误：找不到模型文件 `final_model.joblib`。请先运行 `train_model.py` 脚本来生成模型文件。")
else:
    # 将输入值转换为模型需要的格式
    hypertension_value = 1 if hypertension == '有' else 0
    snoring_value = 1 if snoring == '是' else 0

    # 使用一个字典来收集所有输入值
    # 对于未在界面上展示的特征，我们使用一个合理的默认值
    # 【重要】在实际应用中，您应该添加所有对预测有重要影响的特征作为输入项
    input_data = {
        'age': age,
        'sex': 1,  # 默认为男性
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': bmi_calculated, # 使用自动计算的BMI
        'neck_circumference_cm': neck_circumference,
        'neck_height_ratio': neck_circumference / height_cm if height_cm > 0 else 0,
        'neck_length_cm': 14, # 默认值
        'asa_grade': asa_grade,
        'mallampati_grade': 1, # 默认值
        'hypertension': hypertension_value,
        'diabetes': 0, # 默认值
        'heart_disease': 0, # 默认值
        'osahs': 0, # 默认值
        'smoking': 0, # 默认值
        'alcoholism': 0, # 默认值
        'snoring': snoring_value,
        'sleeping_pills': 0, # 默认值
        'base_systolic_bp': 120, # 默认值
        'base_diastolic_bp': 80, # 默认值
        'base_map': 93, # 默认值
        'base_hr': 75, # 默认值
        'base_spo2': 98 # 默认值
    }

    # 转换为DataFrame，并确保列顺序与训练时一致
    input_df = pd.DataFrame([input_data])[feature_names]

    # --- 进行预测并展示 ---
    if st.sidebar.button('计算风险等级'):
        # 预测概率
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        # 定义风险阈值，并给出定性结果
        risk_threshold = 0.5  # 50%作为高低风险的分界线，您可以根据临床需求调整
        risk_level = "高风险" if prediction_proba >= risk_threshold else "低风险"
        
        st.subheader('预测结果')
        
        # 根据风险等级显示不同颜色和信息
        if risk_level == "高风险":
            st.error(f"风险等级: {risk_level}", icon="⚠️")
            st.warning('建议临床医生对该患者给予额外关注。')
        else:
            st.success(f"风险等级: {risk_level}", icon="✅")
            st.info('风险等级较低，请按常规流程处理。')
        
        # 以较小字体显示具体的概率值作为参考
        st.markdown(f"**具体风险概率为: `{prediction_proba:.2%}`**")
    else:
        st.info('请在左侧输入患者信息，然后点击“计算风险等级”按钮。')

