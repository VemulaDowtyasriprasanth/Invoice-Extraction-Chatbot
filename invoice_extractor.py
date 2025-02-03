import streamlit as st
import pandas as pd
from pypdf import PdfReader
import re
from langchain_community.llms import Ollama
import os

# Initialize Llama model
llm = Ollama(model="llama2:latest")

def get_pdf_text(pdf_doc):
    """Extract text from PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extracted_data(pages_data):
    """Extract structured data using Llama"""
    prompt = f"""Please Extract all the following values from this invoice data:
    invoice no., Description, Quantity, date, Unit price, Amount, Total, email, phone number and address.

    Here is the data to extract from:
    {pages_data}

    Please format your response exactly like this example:
    {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'example@email.com','Phone number': '9999999999','Address': 'Sample Address'}}

    Only provide the JSON-like response, no other text.
    """
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        st.error(f"Error in LLM processing: {str(e)}")
        return "{}"

def create_docs(user_pdf_list):
    """Process PDFs and create dataframe"""
    df = pd.DataFrame({
        'Invoice no.': pd.Series(dtype='str'),
        'Description': pd.Series(dtype='str'),
        'Quantity': pd.Series(dtype='str'),
        'Date': pd.Series(dtype='str'),
        'Unit price': pd.Series(dtype='str'),
        'Amount': pd.Series(dtype='str'),
        'Total': pd.Series(dtype='str'),
        'Email': pd.Series(dtype='str'),
        'Phone number': pd.Series(dtype='str'),
        'Address': pd.Series(dtype='str')
    })

    for filename in user_pdf_list:
        try:
            # Extract text from PDF
            raw_data = get_pdf_text(filename)
            st.info(f"Processing {filename.name}...")

            # Get structured data from LLM
            llm_extracted_data = extracted_data(raw_data)

            # Extract dictionary from response
            pattern = r'{(.+)}'
            match = re.search(pattern, llm_extracted_data, re.DOTALL)

            if match:
                extracted_text = match.group(1)
                try:
                    data_dict = eval('{' + extracted_text + '}')
                    df = df._append([data_dict], ignore_index=True)
                    st.success(f"Successfully processed {filename.name}")
                except Exception as e:
                    st.error(f"Error parsing data from {filename.name}: {str(e)}")
            else:
                st.warning(f"No structured data found in {filename.name}")

        except Exception as e:
            st.error(f"Error processing {filename.name}: {str(e)}")

    return df

def main():
    st.set_page_config(page_title="Invoice Extraction Bot", layout="wide")
    
    # Title and description
    st.title("🤖 Invoice Extraction Bot")
    st.markdown("""
    This bot helps you extract data from invoice PDFs using AI. 
    It uses the Llama language model to process and structure the information.
    """)

    # File uploader
    st.subheader("📄 Upload Invoices")
    pdf_files = st.file_uploader(
        "Upload invoice PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files"
    )

    # Process button
    if st.button("🔍 Extract Data", type="primary"):
        if not pdf_files:
            st.warning("Please upload at least one PDF file")
            return

        with st.spinner('Processing invoices... This may take a few moments.'):
            try:
                # Process PDFs
                df = create_docs(pdf_files)
                
                # Display results
                st.subheader("📊 Extracted Data")
                st.dataframe(df)

                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="extracted_invoices.csv",
                    mime="text/csv"
                )
                
                st.success("✅ Processing complete! You can download the results as a CSV file.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Add helpful information
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Upload one or more PDF invoices using the file uploader above
        2. Click the 'Extract Data' button to process the invoices
        3. Review the extracted data in the table
        4. Download the results as a CSV file
        
        Note: The quality of extraction depends on the PDF quality and format.
        """)

if __name__ == '__main__':
    main()