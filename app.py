from flask import Flask, render_template, request
import pandas as pd
import joblib
import math

app = Flask(__name__)


# ==========================================
# Load Cleaned Dataset
# ==========================================

df = pd.read_csv("data/data_Cleaned.csv")


# ==========================================
# Load Model & Preprocessors
# ==========================================

model = joblib.load("models/best_gbr.pkl")
encoder = joblib.load("models/encoders.pkl")
scaler = joblib.load("models/scaler.pkl")


# ==========================================
# Get values for dropdowns from dataset
# ==========================================

companies = sorted(
    df["Company"].unique()
)

type_names = sorted(
    df["TypeName"].unique()
)

operating_systems = sorted(
    df["OpSys"].unique()
)

processors = sorted(
    df["Processor_Type"].unique()
)

gpu_brands = sorted(
    df["Gpu_Brand"].unique()
)


# ==========================================
# Common data for HTML
# ==========================================

def get_form_data():

    return {
        "companies": companies,
        "type_names": type_names,
        "operating_systems": operating_systems,
        "processors": processors,
        "gpu_brands": gpu_brands
    }


# ==========================================
# Home
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        **get_form_data()
    )


# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # ======================================
    # 1. Receive basic information
    # ======================================

    company = request.form["Company"]

    typename = request.form["TypeName"]

    inches = float(
        request.form["Inches"]
    )

    ram = float(
        request.form["Ram"]
    )

    opsys = request.form["OpSys"]

    weight = float(
        request.form["Weight"]
    )


    # ======================================
    # 2. Touchscreen
    # ======================================

    touchscreen = int(
        request.form["IS_Touchscreen"]
    )


    # ======================================
    # 3. Display Type
    # ======================================

    display_type = request.form["Display_Type"]

    if display_type == "standard":

        ips = 0
        retina = 0

    elif display_type == "ips":

        ips = 1
        retina = 0

    elif display_type == "retina":

        ips = 0
        retina = 1

    elif display_type == "ips_retina":

        ips = 1
        retina = 1

    else:

        ips = 0
        retina = 0


    # ======================================
    # 4. Resolution
    # ======================================

    resolution = request.form["Resolution"]

    x_res, y_res = resolution.split("x")

    x_res = float(x_res)
    y_res = float(y_res)


    # ======================================
    # 5. Calculate PPI
    # ======================================

    ppi = math.sqrt(
        x_res ** 2 +
        y_res ** 2
    ) / inches


    # ======================================
    # 6. Processor
    # ======================================

    processor = request.form[
        "Processor_Type"
    ]

    clock_speed = float(
        request.form["Clock_Speed_GHz"]
    )


    # ======================================
    # 7. GPU
    # ======================================

    gpu_brand = request.form[
        "Gpu_Brand"
    ]


    # ======================================
    # 8. Storage
    # ======================================

    storage_type = request.form[
        "Storage_Type"
    ]


    # Default values

    ssd = 0
    hdd = 0
    flash_storage = 0
    hybrid = 0


    # SSD

    if storage_type == "ssd":

        ssd = float(
            request.form.get(
                "SSD",
                0
            )
        )


    # HDD

    elif storage_type == "hdd":

        hdd = float(
            request.form.get(
                "HDD",
                0
            )
        )


    # SSD + HDD

    elif storage_type == "ssd_hdd":

        ssd = float(
            request.form.get(
                "SSD",
                0
            )
        )

        hdd = float(
            request.form.get(
                "HDD",
                0
            )
        )


    # Flash

    elif storage_type == "flash":

        flash_storage = float(
            request.form.get(
                "Flash_Storage",
                0
            )
        )


    # Hybrid

    elif storage_type == "hybrid":

        hybrid = float(
            request.form.get(
                "Hybrid",
                0
            )
        )


    # ======================================
    # 9. Create original model features
    # ======================================

    data = pd.DataFrame([{

        "Company": company,

        "TypeName": typename,

        "Inches": inches,

        "Ram": ram,

        "OpSys": opsys,

        "Weight": weight,

        "IS_Touchscreen": touchscreen,

        "IPS": ips,

        "Retina Display": retina,

        "X_res": x_res,

        "Y_res": y_res,

        "PPI": ppi,

        "Clock_Speed_GHz": clock_speed,

        "Processor_Type": processor,

        "Gpu_Brand": gpu_brand,

        "SSD": ssd,

        "HDD": hdd,

        "Flash_Storage": flash_storage,

        "Hybrid": hybrid

    }])


    # ======================================
    # 10. Categorical columns
    # ======================================

    categorical_cols = [

        "Company",

        "TypeName",

        "OpSys",

        "Processor_Type",

        "Gpu_Brand"

    ]


    # ======================================
    # 11. Numerical columns
    # ======================================

    numerical_cols = [

        "Inches",

        "Ram",

        "Weight",

        "X_res",

        "Y_res",

        "PPI",

        "Clock_Speed_GHz",

        "SSD",

        "HDD",

        "Flash_Storage",

        "Hybrid"

    ]


    # ======================================
    # 12. Encoding
    # ======================================

    encoded = encoder.transform(
        data[categorical_cols]
    )


    encoded_df = pd.DataFrame(

        encoded,

        columns=encoder.get_feature_names_out(
            categorical_cols
        ),

        index=data.index

    )


    # ======================================
    # 13. Scaling
    # ======================================

    scaled = scaler.transform(
        data[numerical_cols]
    )


    scaled_df = pd.DataFrame(

        scaled,

        columns=numerical_cols,

        index=data.index

    )

    binary_df = data[
    ["IS_Touchscreen", "IPS", "Retina Display"]
    ]
    # ======================================
    # 14. Combine
    # ======================================

    final_data = pd.concat(

        [
            scaled_df,
            encoded_df,
            binary_df
        ],

        axis=1

    )

    final_data = final_data[
    model.feature_names_in_
    ]

    # ======================================
    # 15. Prediction
    # ======================================

    prediction = model.predict(
        final_data
    )[0]


    # ======================================
    # 16. Return result
    # ======================================

    return render_template(

        "index.html",

        **get_form_data(),

        prediction=f"$ {prediction:,.0f}"

    )


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )