import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="AI Car Price Dashboard",
    page_icon="🚗",
    layout="wide" 
)

# Custom CSS for Premium Academic Branding & Advanced Icon Layouts
st.markdown("""
    <style>
    /* العنوان الرئيسي بالألوان الأكاديمية المعتمدة */
    .main-title {
        font-size: 36px;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* ستايل العناوين الفرعية مع الأيقونات */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .icon-wrapper {
        background: #EFF6FF;
        color: #1E3A8A;
        padding: 6px 10px;
        border-radius: 8px;
        margin-right: 10px;
        font-size: 18px;
        font-weight: normal;
    }
    
    /* كروت المؤشرات الرقمية بديلة الـ st.metric العادية لشكل أكثر احترافية */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.1);
    }
    .card-icon {
        font-size: 26px;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 13px;
        color: #4B5563;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* ستايل لوحة البيانات الفنية */
    .info-box {
        background-color: #F8FAFC;
        border-left: 5px solid #1E3A8A;
        padding: 20px;
        border-radius: 12px;
        color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header
st.markdown('<div class="main-title">AI-Driven Vehicle Analytics & Price Predictor</div>', unsafe_allow_html=True)

st.write("---")

# 3. Load Trained Assets
@st.cache_resource
def load_assets():
    model = joblib.load('car_price_best_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Model assets not found. Please check your deployment directory.")
    st.stop()

# --- SMART HIERARCHICAL MAPPING DATA ---
brand_models_dict = {
    "Suzuki": ["Swift", "Dzire", "Alto", "Ritz", "Sx4", "Ciaz", "Ertiga"],
    "Toyota": ["Corolla", "Fortuner", "Innova", "Etios"],
    "Honda": ["City", "Brio", "Amaze"],
    "Hyundai": ["Verna", "i20", "Grand i10", "i10", "Xcent", "Eon"],
    "Other Brands": ["Other Vehicles"]
}

# 4. SIDEBAR - For Advanced/Technical Specifications (تنظيم وتوفير مساحة)
st.sidebar.markdown("<h3 style='color:#1E3A8A;'>⚙️ Advanced Specifications</h3>", unsafe_allow_html=True)
st.sidebar.write("Technical features used by the ML model:")

# تمصير وتبسيط الاختيارات للمستخدم مع الحفاظ على القيمة البرمجية في الخلفية
fuel_user = st.sidebar.selectbox("Fuel Type (نوع الوقود)", ["Petrol (بنزين)", "Diesel (ديزل/سولار)", "CNG (غاز طبيعي)"])
fuel_type = fuel_user.split()[0] # بياخد الكلمة الإنجليزي بس للموديل

transmission_user = st.sidebar.selectbox("Transmission (نوع الفتيس)", ["Automatic (أوتوماتيك)", "Manual (مانيوال)"])
transmission = transmission_user.split()[0]

seller_user = st.sidebar.selectbox("Seller Type (نوع البائع)", ["Dealer (معرض/موزع)", "Individual (فرد مستقل)"])
seller_type = seller_user.split()[0]

owner = st.sidebar.selectbox("Previous Owners (عدد الملاك السابقين)", [0, 1, 3])

# 5. MAIN PAGE LAYOUT - For Primary inputs
st.markdown("""
    <div class="section-header">
        <span class="icon-wrapper">📋</span>
        <span>Primary Car Specifications</span>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    car_brand = st.selectbox("Car Manufacturer / Brand", list(brand_models_dict.keys()))
    available_models = brand_models_dict[car_brand]
    car_model = st.selectbox("Car Model", available_models)

with col2:
    present_price_egp = st.number_input(
        "Present Price of the Car New (in EGP, e.g., 500,000)", 
        min_value=10000, max_value=15000000, value=600000, step=20000
    )
    manufacturing_year = st.number_input("Manufacturing Year (سنة الصنع)", min_value=2000, max_value=2026, value=2018, step=1)

st.write("")

# 6. Prediction and Analytics Engine
if st.button("🔮 Run AI Valuation & Generate Report", type="primary", use_container_width=True):
    
    # Preprocessing
    car_age = 2026 - manufacturing_year
    mapped_transmission = 1 if transmission == "Manual" else 0
    mapped_seller_type = 1 if seller_type == "Dealer" else 0
    
    # Currency Conversion (EGP to Lakhs)
    INR_TO_EGP_RATE = 0.58
    LAKH_TO_EGP = 100000 * INR_TO_EGP_RATE
    present_price_lakhs = present_price_egp / LAKH_TO_EGP
    
    # Prepare DataFrame
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    input_df['Present_Price'] = present_price_lakhs
    input_df['Kms_Driven'] = 30000 # قيمة افتراضية متوسطة أو يمكن إعادتها كمدخل إن أردتِ
    input_df['Owner'] = owner
    input_df['Transmission'] = mapped_transmission
    input_df['Seller_Type'] = mapped_seller_type
    input_df['Car_Age'] = car_age
    
    fuel_column = f"Fuel_Type_{fuel_type}"
    if fuel_column in input_df.columns:
        input_df[fuel_column] = 1
        
    selected_brand = car_brand.lower().split()[0]
    brand_column = f"Brand_{selected_brand}"
    if brand_column in input_df.columns:
        input_df[brand_column] = 1
    
    # ML Prediction
    predicted_price_lakhs = model.predict(input_df)[0]
    
    # Currency Conversion (Lakhs to EGP)
    price_in_inr = predicted_price_lakhs * 100000
    price_in_egp = price_in_inr * INR_TO_EGP_RATE
    if price_in_egp < 0: price_in_egp = 0
    if price_in_egp > present_price_egp: price_in_egp = present_price_egp * 0.9 # منطق سوقي لحماية الحدود
    
    # Calculate Financial Metrics
    depreciation_value = present_price_egp - price_in_egp
    depreciation_rate = (depreciation_value / present_price_egp) * 100
    
    st.write("---")
    st.markdown("""
        <div class="section-header">
            <span class="icon-wrapper">📊</span>
            <span>Valuation Analytics Report</span>
        </div>
    """, unsafe_allow_html=True)
    
    # عرض الكروت المودرن المظبوطة بالأيقونات الملونة والألوان الأكاديمية المعتمدة
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #10B981;">
                <div class="card-icon">💰</div>
                <div class="metric-label">Estimated Resale Value (سعر المستعمل المتوقع)</div>
                <div class="metric-value" style="color: #10B981;">{price_in_egp:,.0f} EGP</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #EF4444;">
                <div class="card-icon">📉</div>
                <div class="metric-label">Value Lost (قيمة الإهلاك بالجنيه)</div>
                <div class="metric-value" style="color: #EF4444;">{depreciation_value:,.0f} EGP <span style="font-size:14px; font-weight:normal;">(-{depreciation_rate:.1f}%)</span></div>
            </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #1E3A8A;">
                <div class="card-icon">⏳</div>
                <div class="metric-label">Car Age (عمر السيارة الحالي)</div>
                <div class="metric-value" style="color: #1E3A8A;">{car_age} Years</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # إضافة الجانب البصري (الشكل والرسم البياني)
    chart_col1, chart_col2 = st.columns([1.2, 1])
    
    with chart_col1:
        st.markdown("""
            <div class="section-header" style="font-size:16px;">
                <span class="icon-wrapper" style="font-size:14px; padding:4px 8px;">📈</span>
                <span>Price Depreciation Comparison</span>
            </div>
        """, unsafe_allow_html=True)
        
        # بناء داتا فريم مصغر للرسم البياني
        plot_data = pd.DataFrame({
            'Price (EGP)': [present_price_egp, price_in_egp]
        }, index=['Original Price (الزيرو)', 'Resale Price (المستعمل)'])
        
        # رسم بياني عمودي شيك جداً وبلت إن في ستريمليت
        st.bar_chart(plot_data, use_container_width=True)
        
    with chart_col2:
        st.markdown("""
            <div class="section-header" style="font-size:16px;">
                <span class="icon-wrapper" style="font-size:14px; padding:4px 8px;">📄</span>
                <span>System Diagnostics & Parameters</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>🔹 Vehicle Profile:</strong> {car_brand} ({car_model})<br><br>
            <strong>🔹 Engine Fuel System:</strong> {fuel_user}<br><br>
            <strong>🔹 Gearbox / Transmission:</strong> {transmission_user}<br><br>
            <strong>🔹 Market Segment Channel:</strong> {seller_user}<br><br>
            <hr style="border:0; border-top:1px solid #E2E8F0; margin:10px 0;">
            <small style="color:#64748B;">Data Pipelines: Inputs dynamically weighted and converted at benchmark rate ({INR_TO_EGP_RATE} EGP/INR).</small>
        </div>
        """, unsafe_allow_html=True)

# 7. Footer Academic Credits
st.write("---")
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 12px;'>© 2026 Smart Mosaic & AI Stylization Systems | Automotive Predictive Analytics Module</p>", unsafe_allow_html=True)