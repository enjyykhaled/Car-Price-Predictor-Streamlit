
# 🚗 AI-Driven Vehicle Analytics & Price Predictor

A production-grade, multi-page interactive web application engineered to eliminate market price asymmetry in the secondary automotive market. This platform bridges complex machine learning regression machinery with commercial-grade telemetry diagnostics, making advanced data science pipelines accessible, responsive, and scannable for enterprise evaluation committees and academic boards.

---

## 📌 1. Project Abstract & Scope

In the used automobile ecosystem, asset valuation suffers from severe volatility due to highly non-linear relationships between physical and macroeconomic feature variables. This sub-module establishes a data-driven intelligence matrix that analyzes structural inputs—such as manufacturing timelines, manufacturing heritage, mechanical engineering setups, and fuel system configurations—to calculate instantaneous, precise asset resale evaluations.

The core infrastructure is designed specifically for graduation project defense frameworks, shifting data science pipelines out of static offline experimental notebooks (`Jupyter Notebooks`) and into a responsive, live deployment environment ready for real-time testing.

---

## 🛠️ 2. System Architecture & Currency Pipeline

The application incorporates a sophisticated pipeline designed to handle cross-border data constraints and enforce market-realistic boundary conditions:

* **Cross-Currency Normalization Engine:** The underlying machine learning model was trained on baseline historical benchmarks denominated in Indian Lakhs (INR). To adapt the system to local market dynamics seamlessly, the backend implements a real-time mathematical translation layer that normalizes Egyptian Pounds (EGP) on-the-fly using an anchor coefficient:
  $$\text{Target Price (Lakhs)} = \frac{\text{Input Price (EGP)}}{100,000 \times 0.58}$$
* **Boundary Condition Protection:** To insulate the platform from statistical anomalies or irrational user inputs (e.g., negative depreciation or hyperinflationary outputs), the execution layer enforces strict boundary limits. It caps the maximum predicted valuation at $90\%$ of the vehicle's original retail price and establishes an absolute floor at 0 EGP.

---

## 🧠 3. Machine Learning & Feature Engineering

The valuation matrix relies on an optimized tree-based regression architecture (Gradient Boosting / Random Forest) to map multi-dimensional features into continuous pricing outputs. The feature engineering pipeline processes user inputs interactively:

* **Deterministic Temporal Scaling:** Rather than evaluating raw production years, the system automatically computes the operational lifespan of the asset relative to the project's baseline target year ($2026$):
  $$\text{Car Age} = 2026 - \text{Manufacturing Year}$$
* **Sparse Binary Vector Transformation (One-Hot Encoding):** Categorical user selections are programmatically expanded into binary matrices to maintain strict alignment with the model's expected shape:
  * **Fuel Type Mapping:** Transforms selections into explicit flags (`Fuel_Type_Petrol`, `Fuel_Type_Diesel`, `Fuel_Type_CNG`).
  * **Transmission & Dealer Profiles:** Encodes manual/automatic gearboxes and sales channels (Individual vs. Dealer) silently in the background, minimizing UI friction.

---

## 🎨 4. UI/UX Architecture & Data Visualization

The graphical interface is built entirely via Streamlit, employing a strict visual hierarchy designed to optimize cognitive load during presentation environments:

* **Sidebar Parameter Isolation:** Complex technical configurations and categorical constants (Fuel Type, Transmission, Seller Type, Past Owners) are isolated within the sidebar to keep the primary dashboard scannable.
* **Main Node Architecture:** Focuses the user's attention on primary high-variance drivers: Brand/Model selection, Present Original Price, and Year of Manufacture.
* **Reactive Visual Analytics:** Upon triggering a prediction, the interface populates customized, color-coded metric cards detailing the predicted valuation alongside a dynamic `matplotlib`/`seaborn` visualization comparing original value retention against depreciation curves.

---

## 🛡️ 5. Fault-Tolerant Execution Shield (Simulation Mode)

Engineered for flawless presentation continuity, the application includes an automated architectural fallback layer:
* **Smart Fallback Mode:** If the pre-compiled binary model weights (`car_price_best_model.pkl`) or the structural mapping matrices (`model_columns.pkl`) are detached, missing, or corrupted on the deployment server, the engine intercepts the error before it breaks the UI.
* **Alternative Mathematical Execution:** The system automatically initializes a robust inner regression simulator. This fallback module utilizes embedded mathematical weights and deterministic decay formulas to mirror the actual model's logic with $100\%$ operational uptime, preventing system crashes during live evaluations.

---

## 📂 6. Repository File Tree

```text
├── app.py                     # Primary Interactive Streamlit Application Source Code
├── requirements.txt           # Institutional Dependency Configuration Manifest
├── project.ipynb              # Monolithic Jupyter Notebook for Training, EDA & Modeling
├── car data.csv               # Historical Training Dataset (301 Observation Records)
├── car_price_best_model.pkl   # Serialized Model Weights (Core Predictive Engine)
├── model_columns.pkl          # Vector Array Column Structural Layout Matrix
└── README.md                  # Comprehensive System Documentation and Manual

```

---

## 🚀 7. Local Ingestion & Deployment Guide

To replicate the execution stack within a local development or testing environment, initialize your system terminal and execute the following commands in sequence:

1. **Clone the Version Matrix:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```


2. **Ingest System Dependencies:**
Ensure your Python virtual environment is activated, then process the dependency manifest:
```bash
pip install -r requirements.txt

```


3. **Execute the Web Engine:**
Boot up the local reactive server instance via Streamlit:
```bash
streamlit run app.py

```



---

## 🎓 8. Academic Credentials & Attribution

* **Academic System Domain:** Core Technical Valuation Module for Senior Graduation Project.
* **Project Baseline Horizon:** 2026.
* **Lead System Developer:** Engy Khaled
* **Specialization Track:** Department of Computer Science & Artificial Intelligence.

```

```
