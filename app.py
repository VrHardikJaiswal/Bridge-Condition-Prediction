
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Bridge Condition Prediction",
    page_icon="🏗️",
    layout="wide"
)

st.title("Bridge Condition Prediction System")

st.write(
    "An Artificial Neural Network-based application "
    "for predicting bridge-condition categories."
)

st.divider()


# ==================================================
# 2. LOAD TRAINED MODEL PACKAGE
# ==================================================

@st.cache_resource
def load_model():

    package = joblib.load("best_bridge_condition_model.sav")

    return package


try:

    package = load_model()

    model = package["model"]
    scaler = package["scaler"]
    feature_columns = package["feature_columns"]
    label_encoder = package["label_encoder"]
    category_info = package["category_info"]

except Exception as e:

    st.error(
        "Unable to load the trained model. "
        "Check that best_bridge_condition_model.sav "
        "is present in the application directory and "
        "contains the required preprocessing information."
    )

    st.exception(e)
    st.stop()


# ==================================================
# 3. DEFINE INPUT VARIABLES
# ==================================================

numeric_columns = [
    "LAT_016",
    "LONG_017",
    "YEAR_BUILT_027",
    "STRUCTURE_LEN_MT_049",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109"
]

categorical_columns = [
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B"
]

required_columns = numeric_columns + categorical_columns


# ==================================================
# 4. PREPARE UNSEEN DATA
# ==================================================

def prepare_data(raw_data):

    encoded_rows = []

    for _, row in raw_data.iterrows():

        encoded_row = {
            column: 0 for column in feature_columns
        }

        # Numerical variables
        for column in numeric_columns:

            value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            if pd.isna(value) or not np.isfinite(value):
                raise ValueError(
                    f"Invalid numerical value in {column}."
                )

            encoded_row[column] = float(value)

        # Categorical variables
        for column in categorical_columns:

            value = str(row[column]).strip()

            info = category_info[column]

            if value not in info["values"]:

                raise ValueError(
                    f"Unknown category '{value}' in {column}. "
                    f"Accepted values: {info['values']}"
                )

            if value != info["baseline"]:

                dummy_column = info["mapping"][value]

                encoded_row[dummy_column] = 1

        encoded_rows.append(encoded_row)

    return pd.DataFrame(
        encoded_rows,
        columns=feature_columns
    )


# ==================================================
# 5. PREDICTION FUNCTION
# ==================================================

def predict_bridges(raw_data):

    prepared_data = prepare_data(raw_data)

    scaled_data = scaler.transform(prepared_data)

    predictions = model.predict(scaled_data)

    predicted_labels = label_encoder.inverse_transform(
        predictions.astype(int)
    )

    probabilities = model.predict_proba(scaled_data)

    confidence = np.max(probabilities, axis=1) * 100

    results = raw_data.copy()

    results["PREDICTED_BRIDGE_CONDITION"] = predicted_labels

    results["PREDICTION_CONFIDENCE (%)"] = np.round(
        confidence, 2
    )

    return results


# ==================================================
# 6. APPLICATION TABS
# ==================================================

tab1, tab2 = st.tabs([
    "Single Bridge Prediction",
    "Batch CSV Prediction"
])


# ==================================================
# TAB 1: SINGLE BRIDGE PREDICTION
# ==================================================

with tab1:

    st.subheader("Predict Condition of an Individual Bridge")

    with st.form("single_bridge_form"):

        col1, col2 = st.columns(2)

        with col1:

            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=21.0
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=72.0
            )

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2026,
                value=2000,
                step=1
            )

        with col2:

            structure_length = st.number_input(
                "Structure Length (metres)",
                min_value=0.0,
                value=100.0
            )

            adt = st.number_input(
                "Average Daily Traffic (ADT)",
                min_value=0.0,
                value=10000.0
            )

            truck_percentage = st.number_input(
                "Truck Traffic Percentage",
                min_value=0.0,
                max_value=100.0,
                value=10.0
            )

        structure_kind = st.selectbox(
            "Structure Kind",
            category_info["STRUCTURE_KIND_043A"]["values"]
        )

        structure_type = st.selectbox(
            "Structure Type",
            category_info["STRUCTURE_TYPE_043B"]["values"]
        )

        submitted = st.form_submit_button(
            "Predict Bridge Condition"
        )

    if submitted:

        input_data = pd.DataFrame([{

            "LAT_016": latitude,
            "LONG_017": longitude,
            "YEAR_BUILT_027": year_built,
            "STRUCTURE_LEN_MT_049": structure_length,
            "ADT_029": adt,
            "PERCENT_ADT_TRUCK_109": truck_percentage,
            "STRUCTURE_KIND_043A": structure_kind,
            "STRUCTURE_TYPE_043B": structure_type

        }])

        try:

            result = predict_bridges(input_data)

            st.success("Prediction completed.")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Bridge Condition",
                str(
                    result[
                        "PREDICTED_BRIDGE_CONDITION"
                    ].iloc[0]
                )
            )

            col2.metric(
                "Prediction Confidence",
                f"{result['PREDICTION_CONFIDENCE (%)'].iloc[0]}%"
            )

            st.subheader("Prediction Details")

            st.dataframe(
                result,
                use_container_width=True
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ==================================================
# TAB 2: BATCH CSV PREDICTION
# ==================================================

with tab2:

    st.subheader("Predict Conditions of Multiple Bridges")

    st.write(
        "Upload a CSV file containing unseen bridge records."
    )

    st.info(
        "The CSV must contain all eight required input variables."
    )

    with st.expander("View required CSV columns"):

        st.code(", ".join(required_columns))

    uploaded_file = st.file_uploader(
        "Upload Bridge Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            unseen_data = pd.read_csv(uploaded_file)

            missing_columns = [
                column for column in required_columns
                if column not in unseen_data.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            elif unseen_data.empty:

                st.error("The uploaded CSV contains no records.")

            else:

                st.subheader("Uploaded Dataset Preview")

                st.dataframe(
                    unseen_data.head(),
                    use_container_width=True
                )

                st.write(
                    f"Total records uploaded: {len(unseen_data)}"
                )

                if st.button("Predict All Bridges"):

                    with st.spinner(
                        "Generating bridge-condition predictions..."
                    ):

                        predictions = predict_bridges(
                            unseen_data
                        )

                    st.success(
                        f"Predictions generated for "
                        f"{len(predictions)} bridges."
                    )

                    st.subheader("Prediction Results")

                    st.dataframe(
                        predictions,
                        use_container_width=True
                    )

                    csv_output = predictions.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_output,
                        file_name="bridge_condition_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Unable to process the uploaded dataset: {e}"
            )


# ==================================================
# 7. MODEL INFORMATION
# ==================================================

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "ANN")

col2.metric("PCA Applied", "No")

col3.metric("Problem Type", "Classification")

st.caption(
    "This application provides analytical predictions "
    "based on historical bridge-condition data. "
    "Predictions are not substitutes for professional "
    "bridge inspections or engineering assessments."
)
