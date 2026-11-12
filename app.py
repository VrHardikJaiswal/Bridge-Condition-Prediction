import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Bridge Condition Assessment",
    page_icon="🌉",
    layout="wide"
)

st.title("Bridge Condition Assessment")
st.write(
    "Predict bridge condition using the saved Artificial Neural Network "
    "trained on the 2025 National Bridge Inventory dataset."
)

MODEL_PATH = "best_bridge_condition_model.sav"

# These are the exact nine columns stored in the uploaded model.
EXPECTED_FEATURES = [
    "LAT_016",
    "LONG_017",
    "YEAR_BUILT_027",
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B",
    "STRUCTURE_LEN_MT_049",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109",
    "DATE_OF_INSPECT_090",
]

# =========================================================
# LOAD THE UPLOADED SAVED MODEL
# =========================================================
@st.cache_resource
def load_model_package():
    return joblib.load(MODEL_PATH)

try:
    package = load_model_package()
    model = package["model"]
    scaler = package["scaler"]
    feature_columns = list(package["feature_columns"])
    label_encoder = package["label_encoder"]
except FileNotFoundError:
    st.error(
        f"Could not find {MODEL_PATH}. Place the .sav file in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

if feature_columns != EXPECTED_FEATURES:
    st.error(
        "The feature columns in this saved model do not match the "
        "nine-feature model this application is designed for.\n\n"
        f"Saved columns: {feature_columns}"
    )
    st.stop()

if model.n_features_in_ != 9 or scaler.n_features_in_ != 9:
    st.error(
        "The ANN and scaler must both accept nine input features. "
        "The uploaded model package does not meet this requirement."
    )
    st.stop()

# =========================================================
# NBI CODE DESCRIPTIONS
# =========================================================
STRUCTURE_KIND_LABELS = {
    0: "Other",
    1: "Concrete",
    2: "Concrete Continuous",
    3: "Steel",
    4: "Steel Continuous",
    5: "Prestressed Concrete",
    6: "Prestressed Concrete Continuous",
    7: "Wood or Timber",
    8: "Masonry",
    9: "Aluminum / Wrought Iron / Cast Iron",
}

STRUCTURE_TYPE_LABELS = {
    0: "Other",
    1: "Slab",
    2: "Stringer / Multi-beam or Girder",
    3: "Girder and Floorbeam System",
    4: "Tee Beam",
    5: "Multiple Box Beam or Girders",
    6: "Single or Spread Box Beam or Girders",
    7: "Frame",
    8: "Orthotropic",
    9: "Deck Truss",
    10: "Through Truss",
    11: "Deck Arch",
    12: "Through Arch",
    13: "Suspension",
    14: "Stayed Girder",
    15: "Movable Lift",
    16: "Movable Bascule",
    17: "Movable Swing",
    18: "Tunnel",
    19: "Culvert",
    20: "Mixed Types",
    21: "Segmental Box Girder",
    22: "Channel Beam",
}

# Categories observed in the uploaded model's training-data diagnostic.
AVAILABLE_KIND_CODES = list(range(10))
AVAILABLE_TYPE_CODES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 19, 20, 21, 22
]

FEATURE_HELP = {
    "Latitude": (
        "Geographical north-south position of the bridge. Enter decimal "
        "degrees. The application converts it to the NBI packed coordinate "
        "format used when the ANN was trained."
    ),
    "Longitude": (
        "Geographical east-west position of the bridge. Enter decimal "
        "degrees. The application converts it to the NBI packed coordinate "
        "format used when the ANN was trained. Enter west longitudes as "
        "negative decimal values; the NBI packed value is stored as positive."
    ),
    "Year Built": (
        "Calendar year in which the bridge was originally constructed."
    ),
    "Structure Kind": (
        "Broad material/construction classification. Select the numerical "
        "NBI code; its meaning is displayed beside the code."
    ),
    "Structure Type": (
        "Main structural design classification. Select the numerical NBI "
        "code; its meaning is displayed beside the code."
    ),
    "Bridge Length": "Total structure length in metres.",
    "Average Daily Traffic": (
        "Average number of vehicles crossing the bridge per day."
    ),
    "Truck Traffic": (
        "Percentage of average daily traffic made up of trucks. Enter 0–100."
    ),
    "Inspection Date": (
        "Date of the inspection represented in the saved model as MMYY. "
        "For example, February 2023 is stored as 223 and October 2019 as 1019."
    ),
}

# =========================================================
# INPUT CONVERSION AND VALIDATION
# =========================================================
def decimal_to_nbi_coordinate(value, max_degrees):
    """
    Convert decimal degrees to the packed NBI DDMMSSss / DDDMMSSss
    numeric representation seen in the model's training data.

    Example:
    40 degrees, 02 minutes, 25.16 seconds -> 40022516
    88 degrees, 07 minutes, 13.46 seconds -> 88071346
    """
    value = abs(float(value))

    if value > max_degrees:
        raise ValueError(
            f"Coordinate must not exceed {max_degrees} degrees."
        )

    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    seconds_hundredths = int(round(seconds * 100))

    # Handle rounding at the 60-second / 60-minute boundaries.
    if seconds_hundredths >= 6000:
        seconds_hundredths = 0
        minutes += 1

    if minutes >= 60:
        minutes = 0
        degrees += 1

    if degrees > max_degrees:
        raise ValueError("Coordinate is outside the valid range.")

    return float(
        degrees * 1_000_000
        + minutes * 10_000
        + seconds_hundredths
    )


def make_inspection_date(month, year):
    """Convert month/year to the numeric MMYY format used by the model."""
    if month < 1 or month > 12:
        raise ValueError("Inspection month must be between 1 and 12.")
    if year < 2000 or year > 2024:
        raise ValueError(
            "Inspection year must be between 2000 and 2024, "
            "matching the range observed in the saved model data."
        )
    return float(month * 100 + (year % 100))


def validate_bridge(data):
    if not -90 <= float(data["latitude"]) <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    if not -180 <= float(data["longitude"]) <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    if not 1800 <= int(data["year_built"]) <= 2025:
        raise ValueError("Enter a valid construction year (1800–2025).")

    if int(data["structure_kind"]) not in AVAILABLE_KIND_CODES:
        raise ValueError("Unsupported Structure Kind code.")

    if int(data["structure_type"]) not in AVAILABLE_TYPE_CODES:
        raise ValueError("Unsupported Structure Type code.")

    if float(data["bridge_length"]) <= 0:
        raise ValueError("Bridge length must be greater than zero.")

    if float(data["adt"]) < 0:
        raise ValueError("Average Daily Traffic cannot be negative.")

    if not 0 <= float(data["truck_percentage"]) <= 100:
        raise ValueError("Truck traffic must be between 0 and 100 percent.")


def prepare_model_input(data):
    validate_bridge(data)

    row = {
        "LAT_016": decimal_to_nbi_coordinate(data["latitude"], 90),
        "LONG_017": decimal_to_nbi_coordinate(data["longitude"], 180),
        "YEAR_BUILT_027": float(data["year_built"]),
        "STRUCTURE_KIND_043A": float(data["structure_kind"]),
        "STRUCTURE_TYPE_043B": float(data["structure_type"]),
        "STRUCTURE_LEN_MT_049": float(data["bridge_length"]),
        "ADT_029": float(data["adt"]),
        "PERCENT_ADT_TRUCK_109": float(data["truck_percentage"]),
        "DATE_OF_INSPECT_090": make_inspection_date(
            int(data["inspection_month"]),
            int(data["inspection_year"]),
        ),
    }

    return pd.DataFrame([row], columns=feature_columns)


def predict_bridge(data):
    model_input = prepare_model_input(data)
    scaled_input = scaler.transform(model_input)
    encoded_prediction = model.predict(scaled_input)
    predicted_label = label_encoder.inverse_transform(
        np.asarray(encoded_prediction, dtype=int)
    )[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        confidence = float(np.max(probabilities) * 100)

    return str(predicted_label), confidence


# =========================================================
# RESULT DISPLAY
# =========================================================
CONDITION_INFO = {
    "G": (
        "Good Condition",
        "The model classifies this bridge as Good. Continue routine "
        "inspection and preventive maintenance."
    ),
    "F": (
        "Fair Condition",
        "The model classifies this bridge as Fair. Review inspection "
        "findings and assess maintenance needs."
    ),
    "P": (
        "Poor Condition",
        "The model classifies this bridge as Poor. A qualified bridge "
        "engineer should review the structure and determine appropriate "
        "inspection, rehabilitation, or replacement actions."
    ),
}


def display_result(prediction, confidence):
    title, description = CONDITION_INFO.get(
        prediction,
        (f"Predicted class: {prediction}", "See the model's class definition.")
    )

    st.subheader("Prediction Result")

    if prediction == "G":
        st.success(title)
    elif prediction == "F":
        st.warning(title)
    elif prediction == "P":
        st.error(title)
    else:
        st.info(title)

    st.write(description)

    if confidence is not None:
        st.metric("Model confidence", f"{confidence:.2f}%")

    st.caption(
        "Model confidence is not a certified probability of structural "
        "safety. This prediction is not a substitute for an engineering "
        "inspection or official bridge-condition assessment."
    )


# =========================================================
# MAIN APPLICATION TABS
# =========================================================
single_tab, batch_tab, guide_tab = st.tabs(
    ["Single Bridge Prediction", "Batch CSV Prediction", "Feature Guide"]
)

# ---------------------------------------------------------
# SINGLE BRIDGE
# ---------------------------------------------------------
with single_tab:
    st.header("Predict the condition of one bridge")

    with st.form("single_bridge_form"):
        st.subheader("Location")

        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "Latitude (decimal degrees)",
                min_value=-90.0,
                max_value=90.0,
                value=40.0,
                format="%.6f",
                help=FEATURE_HELP["Latitude"],
            )
        with col2:
            longitude = st.number_input(
                "Longitude (decimal degrees)",
                min_value=-180.0,
                max_value=180.0,
                value=-88.0,
                format="%.6f",
                help=FEATURE_HELP["Longitude"],
            )

        st.subheader("Construction and structure")

        col1, col2 = st.columns(2)
        with col1:
            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2025,
                value=2000,
                step=1,
                help=FEATURE_HELP["Year Built"],
            )
            bridge_length = st.number_input(
                "Bridge Length (metres)",
                min_value=0.1,
                value=100.0,
                step=1.0,
                help=FEATURE_HELP["Bridge Length"],
            )
        with col2:
            kind_code = st.selectbox(
                "Structure Kind",
                AVAILABLE_KIND_CODES,
                format_func=lambda x: f"{x} — {STRUCTURE_KIND_LABELS[x]}",
                help=FEATURE_HELP["Structure Kind"],
            )
            type_code = st.selectbox(
                "Structure Type",
                AVAILABLE_TYPE_CODES,
                format_func=lambda x: f"{x} — {STRUCTURE_TYPE_LABELS[x]}",
                help=FEATURE_HELP["Structure Type"],
            )

        st.subheader("Traffic characteristics")

        col1, col2 = st.columns(2)
        with col1:
            adt = st.number_input(
                "Average Daily Traffic (vehicles/day)",
                min_value=0,
                value=5000,
                step=100,
                help=FEATURE_HELP["Average Daily Traffic"],
            )
        with col2:
            truck_percentage = st.number_input(
                "Truck Traffic (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help=FEATURE_HELP["Truck Traffic"],
            )

        st.subheader("Inspection date")

        col1, col2 = st.columns(2)
        with col1:
            inspection_month = st.selectbox(
                "Inspection Month",
                list(range(1, 13)),
                format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
                help=FEATURE_HELP["Inspection Date"],
            )
        with col2:
            inspection_year = st.selectbox(
                "Inspection Year",
                list(range(2024, 1999, -1)),
                index=5,  # default 2019
                help=FEATURE_HELP["Inspection Date"],
            )

        submitted = st.form_submit_button(
            "Predict Bridge Condition",
            use_container_width=True,
        )

    if submitted:
        input_data = {
            "latitude": latitude,
            "longitude": longitude,
            "year_built": year_built,
            "structure_kind": kind_code,
            "structure_type": type_code,
            "bridge_length": bridge_length,
            "adt": adt,
            "truck_percentage": truck_percentage,
            "inspection_month": inspection_month,
            "inspection_year": inspection_year,
        }

        try:
            prediction, confidence = predict_bridge(input_data)
            display_result(prediction, confidence)
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")


# ---------------------------------------------------------
# BATCH CSV
# ---------------------------------------------------------
with batch_tab:
    st.header("Predict conditions for multiple bridges")

    st.write(
        "Upload a CSV using the template below. Coordinates must be in "
        "decimal degrees. Inspection date is entered as separate month "
        "and year columns."
    )

    template_df = pd.DataFrame({
        "latitude": [40.037544],
        "longitude": [-88.120406],
        "year_built": [1972],
        "structure_kind": [1],
        "structure_type": [2],
        "bridge_length": [18.8],
        "adt": [5000],
        "truck_percentage": [10.0],
        "inspection_month": [3],
        "inspection_year": [2024],
    })

    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="bridge_input_template.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader(
        "Upload bridge records",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)

            required_batch_columns = list(template_df.columns)
            missing_columns = [
                col for col in required_batch_columns
                if col not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing CSV columns: " + ", ".join(missing_columns)
                )
            elif batch_df.empty:
                st.warning("The uploaded CSV contains no records.")
            else:
                st.write("Uploaded records:")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("Run batch predictions"):
                    output_rows = []
                    progress = st.progress(0.0)
                    status = st.empty()

                    for i, row in batch_df.iterrows():
                        record = row.to_dict()

                        try:
                            prediction, confidence = predict_bridge(record)
                            title, _ = CONDITION_INFO.get(
                                prediction,
                                (prediction, "")
                            )
                            output_rows.append({
                                "Predicted_Condition": prediction,
                                "Condition_Description": title,
                                "Confidence_Percentage": (
                                    round(confidence, 2)
                                    if confidence is not None else None
                                ),
                                "Prediction_Status": "Success",
                            })
                        except Exception as exc:
                            output_rows.append({
                                "Predicted_Condition": None,
                                "Condition_Description": None,
                                "Confidence_Percentage": None,
                                "Prediction_Status": str(exc),
                            })

                        progress.progress((i + 1) / len(batch_df))
                        status.write(
                            f"Processed {i + 1} of {len(batch_df)} records"
                        )

                    results_df = pd.concat(
                        [
                            batch_df.reset_index(drop=True),
                            pd.DataFrame(output_rows),
                        ],
                        axis=1,
                    )

                    success_count = (
                        results_df["Prediction_Status"] == "Success"
                    ).sum()

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total records", len(results_df))
                    c2.metric("Successful predictions", int(success_count))
                    c3.metric("Failed predictions", len(results_df) - int(success_count))

                    st.dataframe(results_df, use_container_width=True)

                    st.download_button(
                        "Download prediction results",
                        data=results_df.to_csv(index=False),
                        file_name="bridge_prediction_results.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")


# ---------------------------------------------------------
# FEATURE GUIDE
# ---------------------------------------------------------
with guide_tab:
    st.header("Feature Guide")

    st.subheader("1. Latitude and Longitude")
    st.write(
        "These identify the bridge's geographical location. The form accepts "
        "decimal degrees and converts them to the packed NBI coordinate "
        "representation used by the saved model. The original training "
        "examples include values such as 40022516 for latitude and 88071346 "
        "for longitude."
    )

    st.subheader("2. Year Built")
    st.write(
        "The calendar year in which the bridge was originally constructed. "
        "It is used as a numerical predictor of bridge condition."
    )

    st.subheader("3. Structure Kind — NBI Item 43A")
    st.write(
        "This describes the broad structural material or construction type. "
        "The numerical code is passed directly to the saved ANN."
    )

    kind_table = pd.DataFrame([
        {"Code": code, "Description": label}
        for code, label in STRUCTURE_KIND_LABELS.items()
    ])
    st.dataframe(kind_table, hide_index=True, use_container_width=True)

    st.subheader("4. Structure Type — NBI Item 43B")
    st.write(
        "This describes the bridge's main structural design. The numerical "
        "code is passed directly to the saved ANN."
    )

    type_table = pd.DataFrame([
        {"Code": code, "Description": label}
        for code, label in STRUCTURE_TYPE_LABELS.items()
    ])
    st.dataframe(type_table, hide_index=True, use_container_width=True)

    st.subheader("5. Bridge Length")
    st.write("Total structure length, measured in metres.")

    st.subheader("6. Average Daily Traffic (ADT)")
    st.write(
        "The estimated average number of vehicles crossing the bridge per day."
    )

    st.subheader("7. Percentage of Truck Traffic")
    st.write(
        "The percentage of average daily traffic represented by trucks."
    )

    st.subheader("8. Date of Inspection — NBI Item 90")
    st.write(
        "This feature is included in the uploaded saved model. The model "
        "stores it as MMYY: for example, 223 means February 2023, 823 means "
        "August 2023, 1019 means October 2019, and 1224 means December 2024."
    )

    st.divider()
    st.subheader("Predicted bridge-condition labels")
    st.markdown(
        """
        | Label | Meaning |
        |---|---|
        | G | Good |
        | F | Fair |
        | P | Poor |
        """
    )

    st.caption(
        "The application reports the class predicted by the saved model. "
        "Predictions are for research and educational use and are not an "
        "official bridge inspection or structural-safety determination."
    )


st.divider()
st.caption(
    "Model: saved ANN (MLPClassifier) | Dataset: 2025 National Bridge "
    "Inventory | Input features: 9"
)
