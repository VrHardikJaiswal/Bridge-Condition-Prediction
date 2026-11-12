
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Bridge Condition Assessment",
    page_icon="🌉",
    layout="wide"
)

st.title("Bridge Condition Assessment System")
st.write(
    "An Artificial Neural Network-based application for "
    "predicting bridge condition using National Bridge "
    "Inventory (NBI) characteristics."
)

st.divider()


# ==========================================================
# 2. LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("best_bridge_condition_model.sav")


try:
    package = load_model()

    model = package["model"]
    scaler = package["scaler"]
    feature_columns = package["feature_columns"]
    label_encoder = package["label_encoder"]

except Exception as e:
    st.error(f"Unable to load model package: {e}")
    st.stop()


# Check that the saved model matches the intended pipeline.

if len(feature_columns) != 36:
    st.error(
        "The saved model does not match the intended "
        "36-feature pipeline.\n\n"
        f"Features found: {len(feature_columns)}\n\n"
        "Please rerun Step 17, the subsequent training "
        "and scaling cells, and the model-saving cell. "
        "Then upload the newly generated .sav file."
    )
    st.stop()

if model.n_features_in_ != len(feature_columns):
    st.error("ANN input dimension does not match saved feature columns.")
    st.stop()

if scaler.n_features_in_ != len(feature_columns):
    st.error("Scaler input dimension does not match saved feature columns.")
    st.stop()


# ==========================================================
# 3. FEATURE INFORMATION
# ==========================================================

NUMERIC_FEATURES = [
    "LAT_016",
    "LONG_017",
    "YEAR_BUILT_027",
    "STRUCTURE_LEN_MT_049",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109"
]

STRUCTURE_KIND = "STRUCTURE_KIND_043A"
STRUCTURE_TYPE = "STRUCTURE_TYPE_043B"

FEATURE_DESCRIPTIONS = {
    "LAT_016": (
        "Geographical latitude of the bridge. "
        "Identifies its north-south position."
    ),

    "LONG_017": (
        "Geographical longitude of the bridge. "
        "Identifies its east-west position."
    ),

    "YEAR_BUILT_027": (
        "Year in which the bridge was originally constructed. "
        "Helps represent the age of the structure."
    ),

    "STRUCTURE_KIND_043A": (
        "Broad classification of the bridge based on its "
        "primary structural material or construction."
    ),

    "STRUCTURE_TYPE_043B": (
        "Classification of the bridge's main structural design, "
        "such as slab, girder, arch or truss."
    ),

    "STRUCTURE_LEN_MT_049": (
        "Total length of the bridge, measured in metres."
    ),

    "ADT_029": (
        "Average Daily Traffic: estimated average number "
        "of vehicles crossing the bridge per day."
    ),

    "PERCENT_ADT_TRUCK_109": (
        "Percentage of average daily traffic consisting "
        "of trucks."
    )
}


# ==========================================================
# 4. NBI STRUCTURE KIND CODE DESCRIPTIONS
# ==========================================================

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
    9: "Aluminum / Wrought Iron / Cast Iron"
}


# ==========================================================
# 5. NBI STRUCTURE TYPE CODE DESCRIPTIONS
# ==========================================================

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
    22: "Channel Beam"
}


# ==========================================================
# 6. DETERMINE CATEGORIES PRESENT IN TRAINING DATA
# ==========================================================

def get_available_codes(column, labels):

    prefix = column + "_"

    codes = []

    for feature in feature_columns:
        if feature.startswith(prefix):
            suffix = feature[len(prefix):]

            try:
                codes.append(int(suffix))
            except ValueError:
                pass

    # The first category is omitted by drop_first=True.
    # The training data uses code 0 as its baseline.
    codes.append(0)

    return sorted(set(
        code for code in codes if code in labels
    ))


kind_codes = get_available_codes(
    STRUCTURE_KIND, STRUCTURE_KIND_LABELS
)

type_codes = get_available_codes(
    STRUCTURE_TYPE, STRUCTURE_TYPE_LABELS
)


# ==========================================================
# 7. FEATURE PREPARATION
# ==========================================================

def prepare_input(data):

    prepared = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical predictors
    for col in NUMERIC_FEATURES:

        if col not in feature_columns:
            raise ValueError(
                f"Required numerical feature missing: {col}"
            )

        prepared.loc[0, col] = float(data[col])

    # Convert decimal coordinates to NBI microdegree format.
    prepared.loc[0, "LAT_016"] = (
        float(data["LAT_016"]) * 1000000
    )

    prepared.loc[0, "LONG_017"] = (
        float(data["LONG_017"]) * 1000000
    )

    # One-hot encode Structure Kind using the training columns.
    kind = int(data[STRUCTURE_KIND])

    if kind not in kind_codes:
        raise ValueError("Structure Kind not present in training data.")

    kind_column = f"{STRUCTURE_KIND}_{kind}"

    if kind != 0:
        if kind_column not in feature_columns:
            raise ValueError(f"Missing model feature: {kind_column}")

        prepared.loc[0, kind_column] = 1

    # One-hot encode Structure Type using the training columns.
    structure_type = int(data[STRUCTURE_TYPE])

    if structure_type not in type_codes:
        raise ValueError("Structure Type not present in training data.")

    type_column = f"{STRUCTURE_TYPE}_{structure_type}"

    if structure_type != 0:
        if type_column not in feature_columns:
            raise ValueError(f"Missing model feature: {type_column}")

        prepared.loc[0, type_column] = 1

    return prepared[feature_columns]


# ==========================================================
# 8. INPUT VALIDATION
# ==========================================================

def validate_input(data):

    latitude = float(data["LAT_016"])
    longitude = float(data["LONG_017"])

    year = int(data["YEAR_BUILT_027"])
    length = float(data["STRUCTURE_LEN_MT_049"])
    adt = float(data["ADT_029"])
    truck = float(data["PERCENT_ADT_TRUCK_109"])

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    if not 1800 <= year <= 2025:
        raise ValueError("Enter a construction year between 1800 and 2025.")

    if length <= 0:
        raise ValueError("Bridge length must be greater than zero.")

    if adt < 0:
        raise ValueError("Average Daily Traffic cannot be negative.")

    if not 0 <= truck <= 100:
        raise ValueError("Truck traffic percentage must be between 0 and 100.")

    if int(data[STRUCTURE_KIND]) not in kind_codes:
        raise ValueError("Invalid Structure Kind code.")

    if int(data[STRUCTURE_TYPE]) not in type_codes:
        raise ValueError("Invalid Structure Type code.")


# ==========================================================
# 9. PREDICTION FUNCTION
# ==========================================================

def predict_bridge(data):

    validate_input(data)

    input_data = prepare_input(data)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    condition = label_encoder.inverse_transform(
        [int(prediction)]
    )[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(np.max(probabilities)) * 100

    return condition, confidence


# ==========================================================
# 10. CONDITION INTERPRETATION
# ==========================================================

CONDITION_DETAILS = {
    "G": {
        "name": "Good Condition",
        "description": (
            "The predicted bridge condition is Good. "
            "Routine inspection and preventive maintenance "
            "remain important."
        )
    },

    "F": {
        "name": "Fair Condition",
        "description": (
            "The predicted bridge condition is Fair. "
            "Further engineering inspection may be required "
            "to identify deterioration and maintenance needs."
        )
    },

    "P": {
        "name": "Poor Condition",
        "description": (
            "The predicted bridge condition is Poor. "
            "A detailed engineering assessment should be "
            "considered to determine rehabilitation or "
            "replacement requirements."
        )
    }
}


def show_prediction(condition, confidence):

    condition = str(condition)

    details = CONDITION_DETAILS.get(
        condition,
        {
            "name": f"Predicted Class: {condition}",
            "description": "Refer to the NBI condition classification."
        }
    )

    st.subheader("Prediction Result")

    if condition == "G":
        st.success(details["name"])

    elif condition == "F":
        st.warning(details["name"])

    elif condition == "P":
        st.error(details["name"])

    else:
        st.info(details["name"])

    st.write(details["description"])

    if confidence is not None:
        st.metric(
            "Model Prediction Confidence",
            f"{confidence:.2f}%"
        )

    st.caption(
        "This is a machine-learning prediction, not a "
        "substitute for a professional bridge inspection."
    )


# ==========================================================
# 11. SINGLE BRIDGE PREDICTION
# ==========================================================

tab1, tab2, tab3 = st.tabs([
    "Single Bridge Assessment",
    "Batch Assessment",
    "Feature Guide"
])


with tab1:

    st.header("Assess an Individual Bridge")

    st.write(
        "Enter the bridge characteristics below. "
        "Use the help descriptions to understand each variable."
    )

    with st.form("bridge_prediction_form"):

        st.subheader("Geographical Information")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.0,
                format="%.6f",
                help=FEATURE_DESCRIPTIONS["LAT_016"]
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-88.0,
                format="%.6f",
                help=FEATURE_DESCRIPTIONS["LONG_017"]
            )

        st.subheader("Construction Information")

        col1, col2 = st.columns(2)

        with col1:
            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2025,
                value=2000,
                step=1,
                help=FEATURE_DESCRIPTIONS["YEAR_BUILT_027"]
            )

        with col2:
            bridge_length = st.number_input(
                "Bridge Length (metres)",
                min_value=0.1,
                value=100.0,
                step=1.0,
                help=FEATURE_DESCRIPTIONS["STRUCTURE_LEN_MT_049"]
            )

        st.subheader("Structural Classification")

        kind_options = [
            f"{code} — {STRUCTURE_KIND_LABELS[code]}"
            for code in kind_codes
        ]

        selected_kind = st.selectbox(
            "Structure Kind",
            kind_options,
            help=FEATURE_DESCRIPTIONS[STRUCTURE_KIND]
        )

        type_options = [
            f"{code} — {STRUCTURE_TYPE_LABELS[code]}"
            for code in type_codes
        ]

        selected_type = st.selectbox(
            "Structure Type",
            type_options,
            help=FEATURE_DESCRIPTIONS[STRUCTURE_TYPE]
        )

        st.subheader("Traffic Information")

        col1, col2 = st.columns(2)

        with col1:
            adt = st.number_input(
                "Average Daily Traffic (vehicles/day)",
                min_value=0,
                value=5000,
                step=100,
                help=FEATURE_DESCRIPTIONS["ADT_029"]
            )

        with col2:
            truck_percentage = st.number_input(
                "Truck Traffic (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help=FEATURE_DESCRIPTIONS["PERCENT_ADT_TRUCK_109"]
            )

        submitted = st.form_submit_button(
            "Predict Bridge Condition",
            use_container_width=True
        )

    if submitted:

        try:

            kind_code = int(selected_kind.split(" — ")[0])
            type_code = int(selected_type.split(" — ")[0])

            bridge_data = {
                "LAT_016": latitude,
                "LONG_017": longitude,
                "YEAR_BUILT_027": year_built,
                "STRUCTURE_KIND_043A": kind_code,
                "STRUCTURE_TYPE_043B": type_code,
                "STRUCTURE_LEN_MT_049": bridge_length,
                "ADT_029": adt,
                "PERCENT_ADT_TRUCK_109": truck_percentage
            }

            condition, confidence = predict_bridge(bridge_data)

            show_prediction(condition, confidence)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==========================================================
# 12. BATCH PREDICTION
# ==========================================================

with tab2:

    st.header("Batch Bridge Assessment")

    st.write(
        "Upload a CSV containing multiple bridge records. "
        "The application will predict the condition of each record."
    )

    st.info(
        "Use decimal latitude and longitude. "
        "Use the original numerical NBI codes for Structure Kind "
        "and Structure Type."
    )

    template = pd.DataFrame({
        "LAT_016": [40.022516],
        "LONG_017": [-88.071346],
        "YEAR_BUILT_027": [1972],
        "STRUCTURE_KIND_043A": [1],
        "STRUCTURE_TYPE_043B": [2],
        "STRUCTURE_LEN_MT_049": [18.8],
        "ADT_029": [5000],
        "PERCENT_ADT_TRUCK_109": [10.0]
    })

    st.download_button(
        "Download CSV Template",
        template.to_csv(index=False),
        file_name="bridge_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload bridge dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            required_columns = [
                "LAT_016",
                "LONG_017",
                "YEAR_BUILT_027",
                "STRUCTURE_KIND_043A",
                "STRUCTURE_TYPE_043B",
                "STRUCTURE_LEN_MT_049",
                "ADT_029",
                "PERCENT_ADT_TRUCK_109"
            ]

            missing = [
                col for col in required_columns
                if col not in batch_df.columns
            ]

            if missing:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )

            else:

                st.write("Uploaded dataset preview:")
                st.dataframe(batch_df.head())

                if st.button("Run Batch Predictions"):

                    results = []
                    errors = []

                    progress = st.progress(0)

                    for index, row in batch_df.iterrows():

                        try:

                            condition, confidence = predict_bridge(
                                row.to_dict()
                            )

                            results.append({
                                "Predicted_Condition": condition,
                                "Condition_Description": (
                                    CONDITION_DETAILS.get(
                                        str(condition), {}
                                    ).get("name", str(condition))
                                ),
                                "Confidence_Percentage": (
                                    round(confidence, 2)
                                    if confidence is not None else None
                                ),
                                "Prediction_Status": "Success"
                            })

                        except Exception as e:

                            results.append({
                                "Predicted_Condition": None,
                                "Condition_Description": None,
                                "Confidence_Percentage": None,
                                "Prediction_Status": str(e)
                            })

                        progress.progress(
                            (index + 1) / len(batch_df)
                        )

                    result_df = pd.concat(
                        [
                            batch_df.reset_index(drop=True),
                            pd.DataFrame(results)
                        ],
                        axis=1
                    )

                    successful = (
                        result_df["Prediction_Status"] == "Success"
                    ).sum()

                    st.subheader("Batch Prediction Summary")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Bridges", len(result_df))
                    col2.metric("Successful Predictions", successful)
                    col3.metric("Failed Predictions", len(result_df) - successful)

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Prediction Results",
                        result_df.to_csv(index=False),
                        file_name="bridge_prediction_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Unable to process CSV: {e}")


# ==========================================================
# 13. FEATURE GUIDE
# ==========================================================

with tab3:

    st.header("Understanding the Model Features")

    st.write(
        "The model uses eight independent variables. "
        "Two of these are categorical variables represented "
        "through one-hot encoding during model training."
    )

    for feature, description in FEATURE_DESCRIPTIONS.items():

        with st.expander(feature):

            st.write(description)

            if feature == STRUCTURE_KIND:

                st.markdown("**Structure Kind Codes**")

                for code in kind_codes:
                    st.write(
                        f"**{code}** — {STRUCTURE_KIND_LABELS[code]}"
                    )

            elif feature == STRUCTURE_TYPE:

                st.markdown("**Structure Type Codes**")

                for code in type_codes:
                    st.write(
                        f"**{code}** — {STRUCTURE_TYPE_LABELS[code]}"
                    )

            elif feature == "LAT_016":
                st.write("Unit: Decimal degrees (latitude).")

            elif feature == "LONG_017":
                st.write("Unit: Decimal degrees (longitude).")

            elif feature == "YEAR_BUILT_027":
                st.write("Unit: Calendar year.")

            elif feature == "STRUCTURE_LEN_MT_049":
                st.write("Unit: Metres.")

            elif feature == "ADT_029":
                st.write("Unit: Vehicles per day.")

            elif feature == "PERCENT_ADT_TRUCK_109":
                st.write("Unit: Percentage (0–100).")


    st.divider()

    st.subheader("Bridge Condition Classification")

    st.markdown("""
    | Code | Classification | Interpretation |
    |---|---|---|
    | G | Good | Bridge condition classified as Good |
    | F | Fair | Bridge condition classified as Fair |
    | P | Poor | Bridge condition classified as Poor |
    """)

    st.caption(
        "Condition labels are based on the classes learned by "
        "the model's LabelEncoder. Predictions should be "
        "validated by qualified bridge inspection professionals."
    )


# ==========================================================
# 14. MODEL DISCLAIMER
# ==========================================================

st.divider()

st.caption(
    "Research and educational application based on the "
    "2025 National Bridge Inventory dataset. Predictions "
    "represent model estimates and must not be treated as "
    "official engineering safety assessments or substitutes "
    "for physical bridge inspections."
)
