# import streamlit as st
# import pandas as pd
# from io import BytesIO

# st.set_page_config(page_title="📁 File Convertor & Cleaner" , layout="wide")
# st.title("📁 File Convertor & Cleaner")
# st.write("Uploas your CSV and Excel Files to clean data convert format effortlessley")

# files = st.file_uploader("Upload CVS or Excel Files",type=["csv", "xlsx"],accept_multiple_files=True)

# if files:
#     for file in files:
#         ext = file.name.split(".")[-1]
#         df = pd.read_csv(file) if ext == "csv" else pdf.read_excel(file)

#         st.subheader(f"🔍 {file.name}I- preview")
#         st.dataframe(df.head())

#         if st.checkbox(f"Fill Missing Values - {file.name}"):
#             df.fillna(df.select_dtypes(include="number").mean(),inplace=True)
#             st.success("Missing values filled successfully!")
#             st.dataframe(df.head())

#             select_columns = st.multiselect(f"Select Columns - {file.name}",df.columns,default = df.columns)
#             df = df[select_columns]
#             st.dataframe(df.head())

#             if st.checkbox(f"Show Chart - {file.name}") and not df.select_dtypes(include="number").empty:
#                 st.bar_chart(df.select_dtypes(include="number").iloc[:, :2])

#             format_choice = st.radio(f"Convert {file.name} to:", ["CSV","Excel"], key=file.name)

#             if st.button(f"Download {file.name} as {format_choice}"):
#                 output = BytesIO()
#                 if format_choice == "CSV":
#                     df.to_csv(output, index=False)
#                     mime = "text/csv"
#                     new_name = file.name.replace(ext, "csv")
#                     else:
#                         df.to_excel(output index=False)
#                         mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                         new_name = file.name.replace(ext, "xlsx")
#                     output.seek(0)
#                     st.download_button("Download File",file_name=new_name, data_output, mine=mine, key=file.name)
#             st.success("Processing Complete!")


import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="📁 Advanced File Converter & Cleaner",
    layout="wide",
    page_icon="📊"
)

# Title and description
st.title("📁 Advanced File Converter & Data Cleaner")
st.write("""
Upload your CSV and Excel files to clean, transform, and convert data effortlessly.
This tool supports advanced data cleaning, transformation, and visualization.
""")

# Sidebar for global settings
with st.sidebar:
    st.header("Settings")
    max_display_rows = st.slider("Max rows to display", 5, 100, 10)
    theme = st.selectbox("Chart Theme", ["default", "darkgrid", "whitegrid"])
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload one or more files
    2. Select cleaning/transformation options
    3. Preview changes
    4. Download processed files
    """)

# File uploader
files = st.file_uploader(
    "Upload CSV or Excel Files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if files:
    for file in files:
        # Create expander for each file
        with st.expander(f"📄 {file.name}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"File: {file.name}")
            
            with col2:
                file_size = f"{len(file.getvalue()) / 1024:.2f} KB"
                st.metric("File Size", file_size)
            
            # Detect file encoding for CSV files
            ext = file.name.split(".")[-1].lower()
            
            try:
                if ext == "csv":
                    # Detect encoding
                    rawdata = file.getvalue()
                    result = chardet.detect(rawdata)
                    encoding = result['encoding']
                    
                    # Try reading with detected encoding, fallback to utf-8
                    try:
                        df = pd.read_csv(file, encoding=encoding)
                    except:
                        df = pd.read_csv(file, encoding='utf-8')
                else:
                    df = pd.read_excel(file)
                
                # Basic file info
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Display data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(df.head(max_display_rows))
                
                # Data cleaning options
                st.subheader("🧹 Data Cleaning Options")
                
                cleaning_options = st.multiselect(
                    f"Select cleaning operations for {file.name}",
                    [
                        "Fill missing values",
                        "Drop missing values",
                        "Remove duplicates",
                        "Convert data types",
                        "Rename columns",
                        "Filter rows",
                        "Standardize text",
                        "Date parsing"
                    ],
                    key=f"cleaning_{file.name}"
                )
                
                # Apply selected cleaning operations
                if cleaning_options:
                    if "Fill missing values" in cleaning_options:
                        st.markdown("#### Fill Missing Values")
                        fill_method = st.selectbox(
                            "Fill method",
                            ["Mean", "Median", "Mode", "Specific value", "Forward fill", "Backward fill"],
                            key=f"fill_method_{file.name}"
                        )
                        
                        if fill_method in ["Mean", "Median", "Mode"]:
                            numeric_cols = df.select_dtypes(include=np.number).columns
                            if len(numeric_cols) > 0:
                                for col in numeric_cols:
                                    if fill_method == "Mean":
                                        df[col].fillna(df[col].mean(), inplace=True)
                                    elif fill_method == "Median":
                                        df[col].fillna(df[col].median(), inplace=True)
                                    elif fill_method == "Mode":
                                        df[col].fillna(df[col].mode()[0], inplace=True)
                                st.success(f"Missing values filled with {fill_method.lower()} for numeric columns")
                            else:
                                st.warning("No numeric columns found for filling")
                        elif fill_method == "Specific value":
                            fill_value = st.text_input("Enter fill value", "0")
                            try:
                                fill_value = float(fill_value)
                                df.fillna(fill_value, inplace=True)
                            except ValueError:
                                df.fillna(fill_value, inplace=True)
                            st.success(f"Missing values filled with {fill_value}")
                        else:
                            df.fillna(method='ffill' if fill_method == "Forward fill" else 'bfill', inplace=True)
                            st.success(f"Missing values filled with {fill_method.lower()}")
                        
                        st.dataframe(df.head(max_display_rows))
                    
                    if "Drop missing values" in cleaning_options:
                        st.markdown("#### Drop Missing Values")
                        drop_thresh = st.slider(
                            "Minimum non-NA values to keep row",
                            0, df.shape[1], df.shape[1],
                            key=f"drop_thresh_{file.name}"
                        )
                        df.dropna(thresh=drop_thresh, inplace=True)
                        st.success(f"Rows with less than {drop_thresh} non-NA values dropped")
                        st.dataframe(df.head(max_display_rows))
                    
                    if "Remove duplicates" in cleaning_options:
                        st.markdown("#### Remove Duplicates")
                        subset = st.multiselect(
                            "Columns to consider for duplicates",
                            df.columns,
                            default=df.columns,
                            key=f"dup_cols_{file.name}"
                        )
                        df.drop_duplicates(subset=subset, inplace=True)
                        st.success("Duplicate rows removed")
                        st.dataframe(df.head(max_display_rows))
                    
                    if "Convert data types" in cleaning_options:
                        st.markdown("#### Convert Data Types")
                        type_cols = st.columns(2)
                        with type_cols[0]:
                            to_str = st.multiselect(
                                "Columns to convert to string",
                                df.columns,
                                key=f"to_str_{file.name}"
                            )
                            if to_str:
                                df[to_str] = df[to_str].astype(str)
                        
                        with type_cols[1]:
                            to_num = st.multiselect(
                                "Columns to convert to numeric",
                                df.columns,
                                key=f"to_num_{file.name}"
                            )
                            if to_num:
                                df[to_num] = pd.to_numeric(df[to_num].replace('[^0-9.-]', '', regex=True), errors='coerce')
                        
                        st.success("Data types converted")
                        st.dataframe(df.head(max_display_rows))
                    
                    if "Rename columns" in cleaning_options:
                        st.markdown("#### Rename Columns")
                        rename_dict = {}
                        cols = st.columns(2)
                        for i, col in enumerate(df.columns):
                            with cols[i % 2]:
                                new_name = st.text_input(
                                    f"Rename '{col}' to:",
                                    value=col,
                                    key=f"rename_{col}_{file.name}"
                                )
                                if new_name != col:
                                    rename_dict[col] = new_name
                        
                        if rename_dict:
                            df.rename(columns=rename_dict, inplace=True)
                            st.success("Columns renamed")
                            st.dataframe(df.head(max_display_rows))
                    
                    if "Filter rows" in cleaning_options:
                        st.markdown("#### Filter Rows")
                        filter_col = st.selectbox(
                            "Select column to filter",
                            df.columns,
                            key=f"filter_col_{file.name}"
                        )
                        
                        if pd.api.types.is_numeric_dtype(df[filter_col]):
                            min_val, max_val = st.slider(
                                "Select range",
                                float(df[filter_col].min()),
                                float(df[filter_col].max()),
                                (float(df[filter_col].min()), float(df[filter_col].max())),
                                key=f"range_{file.name}"
                            )
                            df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
                        else:
                            selected_values = st.multiselect(
                                "Select values to keep",
                                df[filter_col].unique(),
                                key=f"values_{file.name}"
                            )
                            if selected_values:
                                df = df[df[filter_col].isin(selected_values)]
                        
                        st.success("Rows filtered")
                        st.dataframe(df.head(max_display_rows))
                    
                    if "Standardize text" in cleaning_options:
                        st.markdown("#### Standardize Text")
                        text_cols = st.multiselect(
                            "Select text columns to clean",
                            df.select_dtypes(include=['object']).columns,
                            key=f"text_cols_{file.name}"
                        )
                        
                        if text_cols:
                            text_options = st.multiselect(
                                "Text cleaning operations",
                                ["Trim whitespace", "Lowercase", "Uppercase", "Remove special chars"],
                                key=f"text_ops_{file.name}"
                            )
                            
                            for col in text_cols:
                                if "Trim whitespace" in text_options:
                                    df[col] = df[col].str.strip()
                                if "Lowercase" in text_options:
                                    df[col] = df[col].str.lower()
                                if "Uppercase" in text_options:
                                    df[col] = df[col].str.upper()
                                if "Remove special chars" in text_options:
                                    df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
                            
                            st.success("Text standardized")
                            st.dataframe(df.head(max_display_rows))
                    
                    if "Date parsing" in cleaning_options:
                        st.markdown("#### Date Parsing")
                        date_cols = st.multiselect(
                            "Select columns to parse as dates",
                            df.columns,
                            key=f"date_cols_{file.name}"
                        )
                        
                        if date_cols:
                            date_format = st.text_input(
                                "Date format (optional)",
                                help="Example formats: %Y-%m-%d, %m/%d/%Y, %d-%b-%y"
                            )
                            for col in date_cols:
                                try:
                                    if date_format:
                                        df[col] = pd.to_datetime(df[col], format=date_format)
                                    else:
                                        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                                except:
                                    st.warning(f"Could not parse {col} as dates")
                            
                            st.success("Dates parsed")
                            st.dataframe(df.head(max_display_rows))
                
                # Data visualization
                st.subheader("📊 Data Visualization")
                
                if st.checkbox(f"Show visualizations for {file.name}"):
                    num_cols = df.select_dtypes(include=np.number).columns
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    date_cols = df.select_dtypes(include=['datetime']).columns
                    
                    if len(num_cols) > 0 or len(cat_cols) > 0 or len(date_cols) > 0:
                        chart_type = st.selectbox(
                            "Chart type",
                            ["Histogram", "Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Heatmap"],
                            key=f"chart_type_{file.name}"
                        )
                        
                        if chart_type == "Histogram" and len(num_cols) > 0:
                            col = st.selectbox("Select numeric column", num_cols)
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], kde=True, ax=ax)
                            st.pyplot(fig)
                        
                        elif chart_type == "Bar Chart":
                            if len(cat_cols) > 0:
                                x_col = st.selectbox("Select categorical column", cat_cols)
                                if len(num_cols) > 0:
                                    y_col = st.selectbox("Select numeric column (optional)", [None] + list(num_cols))
                                else:
                                    y_col = None
                                
                                fig, ax = plt.subplots()
                                if y_col:
                                    sns.barplot(x=x_col, y=y_col, data=df, ax=ax)
                                else:
                                    df[x_col].value_counts().plot(kind='bar', ax=ax)
                                st.pyplot(fig)
                            else:
                                st.warning("No categorical columns found")
                        
                        elif chart_type == "Line Chart":
                            if len(date_cols) > 0 and len(num_cols) > 0:
                                x_col = st.selectbox("Select date column", date_cols)
                                y_col = st.selectbox("Select numeric column", num_cols)
                                
                                fig, ax = plt.subplots()
                                df.plot(x=x_col, y=y_col, kind='line', ax=ax)
                                st.pyplot(fig)
                            else:
                                st.warning("Need both date and numeric columns")
                        
                        elif chart_type == "Scatter Plot":
                            if len(num_cols) >= 2:
                                x_col = st.selectbox("Select X-axis column", num_cols)
                                y_col = st.selectbox("Select Y-axis column", num_cols)
                                
                                fig, ax = plt.subplots()
                                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
                                st.pyplot(fig)
                            else:
                                st.warning("Need at least 2 numeric columns")
                        
                        elif chart_type == "Box Plot":
                            if len(num_cols) > 0:
                                col = st.selectbox("Select numeric column", num_cols)
                                if len(cat_cols) > 0:
                                    group_col = st.selectbox("Group by (optional)", [None] + list(cat_cols))
                                else:
                                    group_col = None
                                
                                fig, ax = plt.subplots()
                                if group_col:
                                    sns.boxplot(x=group_col, y=col, data=df, ax=ax)
                                else:
                                    sns.boxplot(y=col, data=df, ax=ax)
                                st.pyplot(fig)
                            else:
                                st.warning("No numeric columns found")
                        
                        elif chart_type == "Heatmap":
                            if len(num_cols) >= 2:
                                cols = st.multiselect("Select numeric columns", num_cols, default=num_cols[:5])
                                if len(cols) >= 2:
                                    fig, ax = plt.subplots()
                                    sns.heatmap(df[cols].corr(), annot=True, ax=ax)
                                    st.pyplot(fig)
                                else:
                                    st.warning("Select at least 2 columns")
                            else:
                                st.warning("Need at least 2 numeric columns")
                    else:
                        st.warning("No suitable columns found for visualization")
                
                # File conversion
                st.subheader("💾 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    format_choice = st.radio(
                        f"Convert {file.name} to:",
                        ["CSV", "Excel", "JSON", "Pickle"],
                        key=f"format_{file.name}"
                    )
                
                with col2:
                    new_filename = st.text_input(
                        "Output filename",
                        value=f"cleaned_{file.name.split('.')[0]}",
                        key=f"filename_{file.name}"
                    )
                
                if st.button(f"Process and Download {file.name}"):
                    output = BytesIO()
                    
                    try:
                        if format_choice == "CSV":
                            output_filename = f"{new_filename}.csv"
                            df.to_csv(output, index=False)
                            mime = "text/csv"
                        elif format_choice == "Excel":
                            output_filename = f"{new_filename}.xlsx"
                            df.to_excel(output, index=False)
                            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        elif format_choice == "JSON":
                            output_filename = f"{new_filename}.json"
                            df.to_json(output, orient='records')
                            mime = "application/json"
                        elif format_choice == "Pickle":
                            output_filename = f"{new_filename}.pkl"
                            df.to_pickle(output)
                            mime = "application/octet-stream"
                        
                        output.seek(0)
                        st.download_button(
                            "Download File",
                            data=output,
                            file_name=output_filename,
                            mime=mime,
                            key=f"download_{file.name}"
                        )
                        st.success("File processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
else:
    st.info("👆 Upload one or more CSV or Excel files to get started")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Advanced File Converter & Data Cleaner • Made with Streamlit</p>
        <p>Report issues or contribute on GitHub</p>
    </div>
""", unsafe_allow_html=True)