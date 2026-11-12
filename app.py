
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bridge Condition Assessment",
    page_icon="🌉",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("best_bridge_condition_model.sav")

try:
    package = load_model()

    model = package["model"]
    scaler = package["scaler"]
    feature_columns = package["feature_columns"]
    label_encoder = package["label_encoder"]
    category_info = package["category_info"]

except Exception as e:
    st.error(f"Unable to load the prediction model: {e}")
    st.stop()


# --------------------------------------------------
# FEATURE CONFIGURATION
# --------------------------------------------------

NUMERIC_FEATURES = [
    "LAT_016",
    "LONG_017",
    "YEAR_BUILT_027",
    "STRUCTURE_LEN_MT_049",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109"
]

CATEGORICAL_FEATURES = [
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B"
]

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "LAT_016":
        "The bridge's north–south geographical position. "
        "Together with longitude, it helps identify regional patterns.",

    "LONG_017":
        "The bridge's east–west geographical position. "
        "Used with latitude to locate the bridge.",

    "YEAR_BUILT_027":
        "The year in which the bridge was constructed. "
        "Helps the model consider the age of the structure.",

    "STRUCTURE_KIND_043A":
        "The broad material or structural category, "
        "such as concrete, steel or timber.",

    "STRUCTURE_TYPE_043B":
        "The structural design of the bridge, "
        "such as a slab, girder, arch or truss.",

    "STRUCTURE_LEN_MT_049":
        "The total length of the bridge, measured in metres.",

    "ADT_029":
        "Average Daily Traffic: the estimated number of vehicles "
        "crossing the bridge each day.",

    "PERCENT_ADT_TRUCK_109":
        "The percentage of daily traffic consisting of trucks. "
        "This helps indicate exposure to heavy vehicle traffic."
}

DISPLAY_NAMES = {
    "LAT_016": "Latitude",
    "LONG_017": "Longitude",
    "YEAR_BUILT_027": "Year Built",
    "STRUCTURE_KIND_043A": "Structure Kind",
    "STRUCTURE_TYPE_043B": "Structure Type",
    "STRUCTURE_LEN_MT_049": "Bridge Length (metres)",
    "ADT_029": "Average Daily Traffic (vehicles/day)",
    "PERCENT_ADT_TRUCK_109": "Truck Traffic (%)"
}


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_input(data):
    """
    Converts user inputs into the exact feature structure
    expected by the trained ANN model.
    """

    prepared = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numeric features
    for col in NUMERIC_FEATURES:
        prepared.loc[0, col] = float(data[col])

    # Categorical features
    for col in CATEGORICAL_FEATURES:

        selected_value = str(data[col])
        info = category_info[col]

        baseline = str(info["baseline"])
        mapping = info["mapping"]

        if selected_value not in info["values"]:
            raise ValueError(
                f"Unknown value '{selected_value}' for "
                f"{DISPLAY_NAMES[col]}."
            )

        # The baseline category is represented by all zeros.
        if selected_value != baseline:

            dummy_column = mapping.get(selected_value)

            if dummy_column is None:
                raise ValueError(
                    f"Category mapping missing for {selected_value}."
                )

            if dummy_column not in feature_columns:
                raise ValueError(
                    f"Model feature '{dummy_column}' is missing."
                )

            prepared.loc[0, dummy_column] = 1

    return prepared[feature_columns]


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_input(data):

    if not -90 <= float(data["LAT_016"]) <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= float(data["LONG_017"]) <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    if not 1800 <= int(data["YEAR_BUILT_027"]) <= date.today().year:
        raise ValueError("Please enter a valid construction year.")

    if float(data["STRUCTURE_LEN_MT_049"]) <= 0:
        raise ValueError("Bridge length must be greater than zero.")

    if float(data["ADT_029"]) < 0:
        raise ValueError("Daily traffic cannot be negative.")

    if not 0 <= float(data["PERCENT_ADT_TRUCK_109"]) <= 100:
        raise ValueError("Truck traffic percentage must be between 0 and 100.")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_bridge(data):

    validate_input(data)

    prepared = prepare_input(data)

    scaled_data = scaler.transform(prepared)

    prediction = model.predict(scaled_data)

    predicted_class = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]

    probabilities = model.predict_proba(scaled_data)[0]

    confidence = float(np.max(probabilities)) * 100

    return predicted_class, confidence


def explain_condition(condition):

    explanations = {
        "G": (
            "Good",
            "The bridge's condition ratings fall within the Good category."
        ),
        "F": (
            "Fair",
            "The bridge shows a Fair condition rating and may require "
            "continued monitoring and maintenance."
        ),
        "P": (
            "Poor",
            "The bridge falls within the Poor condition category "
            "and may require priority engineering assessment."
        )
    }

    return explanations.get(
        str(condition).upper(),
        (str(condition), "Refer to the relevant bridge inspection records.")
    )


# --------------------------------------------------
# APPLICATION INTERFACE
# --------------------------------------------------

st.title("Bridge Condition Assessment System")

st.write(
    "This application uses a trained Artificial Neural Network (ANN) "
    "to estimate the condition category of a bridge using its "
    "geographical, structural and traffic-related information."
)

st.info(
    "Enter the available bridge information to obtain a predicted "
    "condition category. Each input includes a short explanation "
    "to help you understand what it represents."
)

st.divider()

single_tab, batch_tab = st.tabs(
    ["Single Bridge Assessment", "Multiple Bridges Assessment"]
)


# ==================================================
# SINGLE BRIDGE ASSESSMENT
# ==================================================

with single_tab:

    st.subheader("Assess an Individual Bridge")

    st.write(
        "Enter the details of one bridge below. "
        "Use information from reliable bridge records wherever possible."
    )

    with st.form("single_bridge_form"):

        st.markdown("#### 1. Geographical Information")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=21.0,
                format="%.6f",
                help=FEATURE_DESCRIPTIONS["LAT_016"]
            )
            st.caption(FEATURE_DESCRIPTIONS["LAT_016"])

        with col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=72.0,
                format="%.6f",
                help=FEATURE_DESCRIPTIONS["LONG_017"]
            )
            st.caption(FEATURE_DESCRIPTIONS["LONG_017"])

        st.divider()

        st.markdown("#### 2. Structural Information")

        col1, col2 = st.columns(2)

        with col1:
            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=date.today().year,
                value=2000,
                step=1,
                help=FEATURE_DESCRIPTIONS["YEAR_BUILT_027"]
            )
            st.caption(FEATURE_DESCRIPTIONS["YEAR_BUILT_027"])

        with col2:
            bridge_length = st.number_input(
                "Bridge Length (metres)",
                min_value=0.1,
                value=100.0,
                step=1.0,
                help=FEATURE_DESCRIPTIONS["STRUCTURE_LEN_MT_049"]
            )
            st.caption(FEATURE_DESCRIPTIONS["STRUCTURE_LEN_MT_049"])

        col1, col2 = st.columns(2)

        with col1:
            kind_values = category_info["STRUCTURE_KIND_043A"]["values"]

            structure_kind = st.selectbox(
                "Structure Kind",
                options=kind_values,
                help=FEATURE_DESCRIPTIONS["STRUCTURE_KIND_043A"]
            )
            st.caption(FEATURE_DESCRIPTIONS["STRUCTURE_KIND_043A"])

        with col2:
            type_values = category_info["STRUCTURE_TYPE_043B"]["values"]

            structure_type = st.selectbox(
                "Structure Type",
                options=type_values,
                help=FEATURE_DESCRIPTIONS["STRUCTURE_TYPE_043B"]
            )
            st.caption(FEATURE_DESCRIPTIONS["STRUCTURE_TYPE_043B"])

        st.divider()

        st.markdown("#### 3. Traffic Information")

        col1, col2 = st.columns(2)

        with col1:
            adt = st.number_input(
                "Average Daily Traffic (vehicles/day)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                help=FEATURE_DESCRIPTIONS["ADT_029"]
            )
            st.caption(FEATURE_DESCRIPTIONS["ADT_029"])

        with col2:
            truck_percentage = st.number_input(
                "Truck Traffic (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                help=FEATURE_DESCRIPTIONS["PERCENT_ADT_TRUCK_109"]
            )
            st.caption(FEATURE_DESCRIPTIONS["PERCENT_ADT_TRUCK_109"])

        submitted = st.form_submit_button(
            "Assess Bridge Condition",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "LAT_016": latitude,
            "LONG_017": longitude,
            "YEAR_BUILT_027": year_built,
            "STRUCTURE_KIND_043A": structure_kind,
            "STRUCTURE_TYPE_043B": structure_type,
            "STRUCTURE_LEN_MT_049": bridge_length,
            "ADT_029": adt,
            "PERCENT_ADT_TRUCK_109": truck_percentage
        }

        try:
            condition, confidence = predict_bridge(input_data)

            condition_name, explanation = explain_condition(condition)

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Bridge Condition", condition_name)

            with col2:
                st.metric("Model Confidence", f"{confidence:.2f}%")

            st.write(explanation)

            st.caption(
                "Model confidence represents the highest predicted "
                "class probability. It is not a guarantee of prediction accuracy."
            )

        except Exception as e:
            st.error(f"Unable to generate prediction: {e}")


# ==================================================
# BATCH BRIDGE ASSESSMENT
# ==================================================

with batch_tab:

    st.subheader("Assess Multiple Bridges")

    st.write(
        "Upload a CSV file containing information about multiple bridges. "
        "The application will generate a predicted condition for each record."
    )

    st.markdown("#### Required CSV Columns")

    st.dataframe(
        pd.DataFrame({
            "Column Name": REQUIRED_COLUMNS,
            "Description": [
                FEATURE_DESCRIPTIONS[col]
                for col in REQUIRED_COLUMNS
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The CSV must contain all eight columns with the exact column names shown above."
    )

    uploaded_file = st.file_uploader(
        "Upload Bridge Data (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            missing_columns = [
                col for col in REQUIRED_COLUMNS
                if col not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded file is missing these required columns: "
                    + ", ".join(missing_columns)
                )

            elif batch_df.empty:

                st.warning("The uploaded CSV contains no records.")

            else:

                st.write("Uploaded data preview")
                st.dataframe(
                    batch_df.head(),
                    use_container_width=True
                )

                st.write(f"Total bridge records: {len(batch_df)}")

                if st.button(
                    "Assess All Bridges",
                    use_container_width=True
                ):

                    results = []
                    errors = []

                    progress = st.progress(0)

                    for index, row in batch_df.iterrows():

                        try:
                            input_data = {
                                col: row[col]
                                for col in REQUIRED_COLUMNS
                            }

                            condition, confidence = predict_bridge(input_data)

                            condition_name, explanation = explain_condition(
                                condition
                            )

                            results.append({
                                "Predicted Condition": condition_name,
                                "Model Confidence (%)": round(confidence, 2),
                                "Condition Explanation": explanation
                            })

                        except Exception as e:

                            results.append({
                                "Predicted Condition": "Error",
                                "Model Confidence (%)": np.nan,
                                "Condition Explanation": str(e)
                            })

                            errors.append(index)

                        progress.progress(
                            (index + 1) / len(batch_df)
                        )

                    results_df = pd.DataFrame(results)

                    final_df = pd.concat(
                        [
                            batch_df.reset_index(drop=True),
                            results_df
                        ],
                        axis=1
                    )

                    st.session_state["bridge_results"] = final_df

                    st.success(
                        f"Assessment completed for {len(batch_df)} records."
                    )

                    if errors:
                        st.warning(
                            f"{len(errors)} record(s) could not be assessed. "
                            "Check the explanation in the results."
                        )

        except Exception as e:
            st.error(f"Unable to read the uploaded CSV: {e}")

    if "bridge_results" in st.session_state:

        final_df = st.session_state["bridge_results"]

        st.divider()
        st.subheader("Batch Assessment Results")

        st.dataframe(
            final_df,
            use_container_width=True
        )

        csv_data = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="bridge_condition_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# --------------------------------------------------
# CONDITION CATEGORY GUIDE
# --------------------------------------------------

st.divider()

with st.expander("Understand the Bridge Condition Categories"):

    st.markdown("""
    | Category | Meaning | General interpretation |
    |---|---|---|
    | **Good (G)** | Good condition | Condition ratings fall within the Good category. |
    | **Fair (F)** | Fair condition | Condition ratings fall within the Fair category. |
    | **Poor (P)** | Poor condition | Condition ratings fall within the Poor category. |
    """)

    st.caption(
        "These are model-predicted categories based on the training data. "
        "They are not substitutes for an official bridge inspection or "
        "professional engineering assessment."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Bridge Condition Assessment System | "
    "Powered by a trained Artificial Neural Network (ANN)"
)
