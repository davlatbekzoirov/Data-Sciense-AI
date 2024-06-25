import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import pandas as pd
import deepdoctection as dd
from IPython.core.display import HTML
import random
import matplotlib.pyplot as plt
from PIL import Image
import base64
import os
import pytesseract

def analyze_pdf(pdf_path, ocr=False, lang='eng'):
    analyzer = dd.get_dd_analyzer()
    df = analyzer.analyze(path=pdf_path)
    df.reset_state()
    doc = iter(df)
    page = next(doc)

    image_path = "marked_image.png"
    text_content = page.text.strip()

    if ocr:
        text_content += "\n" + perform_ocr(page.image_path, lang)

    return image_path, text_content

def perform_ocr(image_path, lang='eng'):
    image = plt.imread(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def main(df):
    st.title("PDF Analysis with Streamlit")

    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.sidebar.title("OCR Options")
        ocr_option = st.sidebar.checkbox("Perform OCR")
        lang = st.sidebar.text_input("OCR Language (e.g., 'eng')")

        st.sidebar.title("Output Options")
        output_option = st.sidebar.checkbox("Save as HTML Report")
        if output_option:
            output_path = st.sidebar.text_input("Output HTML Report Path")

        if st.sidebar.button("Analyze PDF"):
            with st.spinner("Analyzing PDF..."):
                if ocr_option:
                    image_path, text_content = analyze_pdf(uploaded_file, True, lang)
                else:
                    image_path, text_content = analyze_pdf(uploaded_file, False)

            st.title("This is the Image")
            image = Image.open(image_path)
            st.image(image, caption="Document Image", use_column_width=True)

            x1 = np.random.randn(200) - 2
            x2 = np.random.randn(200)
            x3 = np.random.randn(200) + 2
            hist_data = [x1, x2, x3]
            st.title("Animation")
            group_labels = ["Document", "Text", "Image"]
            fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])
            st.plotly_chart(fig, use_container_width=True)

            st.title("Table")
            data = {
                "Element Type": ["Document", "Text", "Image"],
                "Count": [len(df), len(text_content), 1],
            }
            df2 = pd.DataFrame(data)
            st.table(df2)

            st.title("This is the Text from the Image")
            st.title(f"{text_content} Length of Text: {len(text_content)}")

            st.title("This is JSON from the Image")
            st.json(
                {
                    "blocks": [
                        {
                            "coordinates": random.sample(
                                [[100, 100], [200, 200], [300, 300], [100, 100]], 4
                            ),
                            "text": "This is a block of text.",
                        },
                        {
                            "coordinates": random.sample(
                                [[400, 400], [500, 500], [600, 600], [400, 400]], 4
                            ),
                            "text": "This is another block of text.",
                        },
                    ],
                    "images": [image_path],
                }
            )

if __name__ == "__main__":
    main()
