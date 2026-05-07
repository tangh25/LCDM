# import streamlit as st
# import pandas as pd
# import joblib
# import os
#
# # --- 1. 页面基本设置 ---
# st.set_page_config(
#     page_title="PTC Metastasis Predictor",
#     layout="wide",
#     initial_sidebar_state="expanded"  # 关键点：强制侧边栏默认展开
# )
#
# # --- 2. 严格对齐模型要求的特征顺序 ---
# # 确保顺序为：subtype, grade, age, Metachronous, 以及 8 个分子特征
# FEATURES = [
#     'subtype', 'grade', 'age', 'Metachronous', 'EGFR_L747_A750del',
#     'EGFR_T790M', 'EGFR_CNV_Duplication', 'EGFR_C797', 'EGFR_S768',
#     'TP53_R248', 'KRAS_G12D', 'KMT2D_mut'
# ]
#
#
# # --- 3. 加载模型 ---
# @st.cache_resource
# def load_model():
#     model_path = 'svm_model.pkl'
#     if os.path.exists(model_path):
#         return joblib.load(model_path)
#     return None
#
#
# model = load_model()
#
# # --- 4. 侧边栏：输入区域 (Sidebar) ---
# # 这里的组件放在 if 按钮 之外，确保它们始终可见
# with st.sidebar:
#     st.header("📝 Patient Data Input")
#     st.markdown("Please select the clinical and molecular features:")
#
#     st.subheader("Clinical Features")
#     val_subtype = st.selectbox("Subtype (0-3):", options=[0, 1, 2, 3], index=1)
#     val_grade = st.selectbox("Tumor Grade (1-4):", options=[1, 2, 3, 4], index=1)
#     val_age = st.number_input("Age (years):", min_value=0, max_value=120, value=55)
#     val_metachronous = st.selectbox("Metachronous (1=Yes, 0=No):", options=[1, 0], index=1)
#
#     st.divider()
#     st.subheader("Molecular Indicators")
#     val_e_del = st.selectbox("EGFR_L747_A750del (1/0):", [1, 0], index=1)
#     val_e_t790m = st.selectbox("EGFR_T790M (1/0):", [1, 0], index=1)
#     val_e_cnv = st.selectbox("EGFR_CNV_Duplication (1/0):", [1, 0], index=1)
#     val_e_c797 = st.selectbox("EGFR_C797 (1/0):", [1, 0], index=1)
#     val_e_s768 = st.selectbox("EGFR_S768 (1/0):", [1, 0], index=1)
#     val_p53 = st.selectbox("TP53_R248 (1/0):", [1, 0], index=1)
#     val_kras = st.selectbox("KRAS_G12D (1/0):", [1, 0], index=1)
#     val_kmt2d = st.selectbox("KMT2D_mut (1/0):", [1, 0], index=1)
#
# # --- 5. 主页面内容 ---
# st.title("Predicting the occurrence of distant metastases after surgery in patients with papillary thyroid cancer")
# st.markdown("---")
#
# # 准备输入数据
# input_dict = {
#     'subtype': val_subtype, 'grade': val_grade, 'age': val_age, 'Metachronous': val_metachronous,
#     'EGFR_L747_A750del': val_e_del, 'EGFR_T790M': val_e_t790m, 'EGFR_CNV_Duplication': val_e_cnv,
#     'EGFR_C797': val_e_c797, 'EGFR_S768': val_e_s768, 'TP53_R248': val_p53,
#     'KRAS_G12D': val_kras, 'KMT2D_mut': val_kmt2d
# }
#
# # 强制按照 FEATURES 列表的顺序排列 DataFrame
# input_df = pd.DataFrame([input_dict])[FEATURES].astype(float)
#
# st.header("Predicting risk:")
#
# # 预测触发按钮
# if st.button("🚀 Calculate Metastasis Risk"):
#     if model is not None:
#         # 进行预测并获取概率
#         prob = model.predict_proba(input_df)[0][1]
#
#         # 结果展示
#         if prob >= 0.5:
#             st.error(f"### This patient is at high risk of distant metastasis!")
#             st.error(f"## The probability of distant metastasis is: {prob:.2%}")
#         else:
#             st.success(f"### This patient is at low risk of distant metastasis.")
#             st.success(f"## The probability of distant metastasis is: {prob:.2%}")
#
#         # 模型解释性展示
#         st.divider()
#         st.markdown("### 🔍 Model Interpretation")
#         st.info(
#             "The prediction is primarily driven by high-impact features such as KRAS_G12D and TP53_R248 based on SHAP analysis.")
#     else:
#         st.error("Error: Model file 'svm_model.pkl' not found.")
# else:
#     # 引导提示
#     st.info(
#         "Please adjust patient parameters in the sidebar and click 'Calculate Metastasis Risk' to view the prediction.")
#
# # --- 6. 底部备注 (与示例图完全一致) ---
# st.markdown("""
# <br><br><br>
# <p style='font-size: 0.85rem; color: gray;'>
# <strong>Note:</strong> a. This website aims to develop and validate a model using machine learning algorithms to predict the risk of distant metastasis in patients with papillary thyroid carcinoma. <br>
# b. By simply inputting the postoperative pathological information and molecular indicators, it is possible to predict the risk of distant metastasis.
# </p>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="Lung Cancer Distant Metastasis Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 特征顺序对齐 ---
FEATURES = [
    'subtype', 'grade', 'age', 'Metachronous', 'EGFR_L747_A750del',
    'EGFR_T790M', 'EGFR_CNV_Duplication', 'EGFR_C797', 'EGFR_S768',
    'TP53_R248', 'KRAS_G12D', 'KMT2D_mut'
]


# --- 3. 模型加载 ---
@st.cache_resource
def load_model():
    model_path = 'svm_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_model()

# --- 4. 优化后的侧边栏排版 ---
with st.sidebar:
    st.title("📝 Data Input")
    st.subheader("Clinical Features")

    # 定义选项映射字典（数字 -> 描述）
    subtype_map = {
        0: "Small Cell Carcinoma",
        1: "Adenocarcinoma",
        2: "Squamous Cell Carcinoma",
        3: "Large Cell Carcinoma",
        5: "Other Neuroendocrine",
        6: "Sarcomatoid Carcinoma"
    }

    grade_map = {
        1: "G1",
        2: "G2",
        3: "G3",
        4: "G4"
    }

    metachronous_map = {
        1: "Yes",
        0: "No"
    }

    # 使用描述性文本显示，但存储原始数字值
    val_subtype = st.selectbox(
        "Subtype:",
        options=list(subtype_map.keys()),
        format_func=lambda x: subtype_map[x],  # 显示映射后的文本
        index=1
    )

    val_age = st.number_input("Age:", 0, 120, 55, help="Patient's age (years old)")

    val_grade = st.selectbox(
        "Grade:",
        options=list(grade_map.keys()),
        format_func=lambda x: grade_map[x],
        index=1
    )

    val_metachronous = st.selectbox(
        "Metachronous:",
        options=list(metachronous_map.keys()),
        format_func=lambda x: metachronous_map[x],
        index=1
    )

    st.divider()

    # 可选：在侧边栏底部显示当前选择的含义
    with st.expander("ℹ️ View current clinical characteristics selection"):
        st.write(f"**Subtype**: {subtype_map[val_subtype]}")
        st.write(f"**Age**: {val_age} years old")
        st.write(f"**Grade**: {grade_map[val_grade]}")
        st.write(f"**Metachronous**: {metachronous_map[val_metachronous]}")

#
#
# with st.sidebar:
#     st.title("📝 Data Input")
#
#     # 临床特征
#     st.subheader("Clinical Features")
#
#     # 方案1：直接移除 columns（推荐，单列无需使用）
#     val_subtype = st.selectbox("Subtype:", [0, 1, 2, 3, 5, 6], index=1)
#     val_age = st.number_input("Age:", 0, 120, 55)
#     val_grade = st.selectbox("Grade:", [1, 2, 3, 4], index=1)
#     val_metachronous = st.selectbox("Metachronous:", [1, 0], index=1)
#
#     st.divider()
# with st.sidebar:
#     st.title("📝 Data Input")
#
#     # 临床特征：紧凑排列
#     st.subheader("Clinical Features")
#     col_c1, col_c2 = st.columns(1)
#     with col_c1:
#         val_subtype = st.selectbox("Subtype:", [0, 1, 2, 3, 5, 6], index=1)
#         val_age = st.number_input("Age:", 0, 120, 55)
#     with col_c2:
#         val_grade = st.selectbox("Grade:", [1, 2, 3, 4], index=1)
#         val_metachronous = st.selectbox("Metachronous:", [1, 0], index=1)
#
#     st.divider()

    # 分子特征：采用两列并行排列，大幅节省垂直空间
    st.subheader("Molecular Indicators (0:No mutation; 1:Mutation)")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        val_e_del = st.selectbox("EGFR_Del:", [1, 0], index=1)
        val_e_cnv = st.selectbox("EGFR_CNV:", [1, 0], index=1)
        val_e_s768 = st.selectbox("EGFR_S768:", [1, 0], index=1)
        val_kras = st.selectbox("KRAS_G12D:", [1, 0], index=1)
    with col_m2:
        val_e_t790m = st.selectbox("EGFR_T790M:", [1, 0], index=1)
        val_e_c797 = st.selectbox("EGFR_C797:", [1, 0], index=1)
        val_p53 = st.selectbox("TP53_R248:", [1, 0], index=1)
        val_kmt2d = st.selectbox("KMT2D_mut:", [1, 0], index=1)

    # 新增：分子特征查看展开块
    with st.expander("ℹ️ View current molecular feature selection"):
        st.write(f"**EGFR_L747_A750del**: {'Yes' if val_e_del == 1 else 'No'}")
        st.write(f"**EGFR_T790M**: {'Yes' if val_e_t790m == 1 else 'No'}")
        st.write(f"**EGFR_CNV_Duplication**: {'Yes' if val_e_cnv == 1 else 'No'}")
        st.write(f"**EGFR_C797**: {'Yes' if val_e_c797 == 1 else 'No'}")
        st.write(f"**EGFR_S768**: {'Yes' if val_e_s768 == 1 else 'No'}")
        st.write(f"**TP53_R248**: {'Yes' if val_p53 == 1 else 'No'}")
        st.write(f"**KRAS_G12D**: {'Yes' if val_kras == 1 else 'No'}")
        st.write(f"**KMT2D_mut**: {'Yes' if val_kmt2d == 1 else 'No'}")

# --- 5. 主页面布局 ---
st.title("Predicting distant metastasis in lung cancer patients")
st.markdown("---")

# 准备数据并预测
input_dict = {
    'subtype': val_subtype, 'grade': val_grade, 'age': val_age, 'Metachronous': val_metachronous,
    'EGFR_L747_A750del': val_e_del, 'EGFR_T790M': val_e_t790m, 'EGFR_CNV_Duplication': val_e_cnv,
    'EGFR_C797': val_e_c797, 'EGFR_S768': val_e_s768, 'TP53_R248': val_p53,
    'KRAS_G12D': val_kras, 'KMT2D_mut': val_kmt2d
}

input_df = pd.DataFrame([input_dict])[FEATURES].astype(float)

st.header("Predicting distant metastatic outcomes:")

if st.button("🚀 Calculate Distant Metastasis Risk"):
    if model is not None:
        prob = model.predict_proba(input_df)[0][1]

        if prob >= 0.5:
            st.error(f"### High Risk: Distant metastasis is likely.")
            st.error(f"## Probability: {prob:.2%}")
        else:
            st.success(f"### Low Risk: Distant metastasis is unlikely.")
            st.success(f"## Probability: {prob:.2%}")

        st.divider()
        st.markdown("### 🔍 Model Interpretation")
        st.info("The prediction is primarily driven by features such as subtype, grade, age, Metachronous, EGFR_L747_A750del,EGFR_T790M, EGFR_CNV_Duplication, EGFR_C797, EGFR_S768,TP53_R248, KRAS_G12D, KMT2D_mut.")
    else:
        st.error("Model file 'svm_model.pkl' not found.")
else:
    st.info("Adjust parameters in the sidebar and click the button to predict.")

# 底部备注
st.markdown("""
<br>
<p style='font-size: 0.8rem; color: gray;'>
<strong>Note:</strong> a. This model uses machine learning to predict distant metastasis risk in lung cancer patients. <br>
b. Accurate prediction requires patients' clinical and molecular indicators.
</p>
""", unsafe_allow_html=True)



