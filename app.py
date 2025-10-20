import streamlit as st
import pandas as pd
import joblib

# --- 页面基础设置 ---
st.set_page_config(page_title="临床风险预测模型", layout="wide")

# --- 加载训练好的模型和阈值 ---
@st.cache_resource
def load_model_and_threshold():
    try:
        # 加载包含模型和阈值的字典
        saved_object = joblib.load('final_model.joblib')
        return saved_object
    except FileNotFoundError:
        return None

saved_data = load_model_and_threshold()
model = None
calculated_threshold = 0.5 # 如果模型文件不存在，则默认为0.5

if saved_data:
    model = saved_data['model']
    calculated_threshold = saved_data['threshold']

# --- 主页面 ---
st.title('术中低氧血症风险预测工具')
st.write("请在下方输入患者的临床指标，然后点击“计算风险等级”按钮。")
st.markdown("---")

# --- 输入区域 ---
st.header("患者信息输入")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider('年龄 (岁)', 18, 100, 50)
    height_cm = st.number_input('身高 (cm)', min_value=100.0, max_value=220.0, value=170.0, step=0.5)
    weight_kg = st.number_input('体重 (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.5)
with col2:
    neck_circumference = st.slider('颈围 (cm)', 20, 60, 38)
    asa_grade = st.selectbox('ASA分级', [1, 2, 3, 4], index=0)
    hypertension = st.selectbox('是否有高血压', ['无', '有'], index=0)
with col3:
    snoring = st.selectbox('是否打鼾', ['否', '是'], index=0)
    if height_cm > 0:
        bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
        st.metric(label="自动计算的BMI值", value=f"{bmi_calculated:.2f}")
    else:
        bmi_calculated = 0
        st.error("身高不能为0")

# --- 风险阈值调整 ---
st.markdown("---")
st.subheader("风险阈值调整")
risk_threshold_percent = st.slider(
    '当预测概率高于此值时，判断为“高风险”：', 
    min_value=1, 
    max_value=99, 
    value=int(calculated_threshold * 100),  # 默认值设为算法计算出的阈值
    format='%d%%'
)
risk_threshold = risk_threshold_percent / 100.0

# --- 按钮和输出区域 ---
st.markdown("---")
_, col_button, _ = st.columns([2, 1, 2])
with col_button:
    calculate_button = st.button('计算风险等级', use_container_width=True)

if model is None:
    st.error("错误：找不到模型文件 `final_model.joblib`。请先运行 `train_model.py` 脚本。")
elif calculate_button:
    # --- 数据准备 ---
    feature_names = [
        'age', 'sex', 'height_cm', 'weight_kg', 'bmi', 'neck_circumference_cm', 'neck_height_ratio', 'neck_length_cm',
        'asa_grade', 'mallampati_grade', 'hypertension', 'diabetes', 'heart_disease', 'osahs', 'smoking',
        'alcoholism', 'snoring', 'sleeping_pills', 'base_systolic_bp', 'base_diastolic_bp', 'base_map',
        'base_hr', 'base_spo2'
    ]
    hypertension_value = 1 if hypertension == '有' else 0
    snoring_value = 1 if snoring == '是' else 0
    input_data = {
        'age': age, 'sex': 1, 'height_cm': height_cm, 'weight_kg': weight_kg,
        'bmi': bmi_calculated, 'neck_circumference_cm': neck_circumference,
        'neck_height_ratio': neck_circumference / height_cm if height_cm > 0 else 0,
        'neck_length_cm': 14, 'asa_grade': asa_grade, 'mallampati_grade': 1,
        'hypertension': hypertension_value, 'diabetes': 0, 'heart_disease': 0,
        'osahs': 0, 'smoking': 0, 'alcoholism': 0, 'snoring': snoring_value,
        'sleeping_pills': 0, 'base_systolic_bp': 120, 'base_diastolic_bp': 80,
        'base_map': 93, 'base_hr': 75, 'base_spo2': 98
    }
    input_df = pd.DataFrame([input_data])[feature_names]

    # --- 预测与展示 ---
    prediction_proba = model.predict_proba(input_df)[0][1]
    risk_level = "高风险" if prediction_proba >= risk_threshold else "低风险"
    
    st.header('预测结果')
    if risk_level == "高风险":
        st.error(f"风险等级: {risk_level}", icon="⚠️")
        st.warning('建议临床医生对该患者给予额外关注。')
    else:
        st.success(f"风险等级: {risk_level}", icon="✅")
        st.info('风险等级较低，请按常规流程处理。')
    
    st.markdown(f"**具体风险概率为: `{prediction_proba:.2%}`** (当前判断阈值为: `{risk_threshold_percent}%`)")

