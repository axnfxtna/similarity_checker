import streamlit as st
import requests
import base64

# Centered title
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("Similarity Checker")

# Upload input
uploaded_pdf_file = st.file_uploader("Upload your search PDF", type=["pdf"])
explanation = st.checkbox("Explanation", value=False)

query_pdf = None
if uploaded_pdf_file:
    query_pdf = base64.b64encode(uploaded_pdf_file.read()).decode("utf-8")

if st.button("Check"):
    if query_pdf is None:
        st.warning("Please upload a PDF before checking.")
    else:
        payload = {"query_pdf": query_pdf, "explanation": explanation}
        endpoint = "http://similarity-app:8005/explanation" if explanation else "http://similarity-app:8005/check"

        try:
            response = requests.post(endpoint, json=payload)
            st.write("Response status:", response.status_code)

            if response.status_code == 200:
                data = response.json()

                # Show explanations (if requested)
                if explanation:
                    st.subheader("Explanation")
                    explanations = data.get("explanations", [])
                    for item in explanations:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write(f"Query Page {item['query_page']}")
                            st.write(f"Attempt {item['attempt']}")
                            st.write(f"🔍 Similarity: {item['similarity']}%")
                        with col2:
                            st.markdown(
                                f"**Matched PDF:** {item['matched_pdf_file']}\n\n"
                                f"**Matched Page:** {item['match_page']}\n\n"
                                f"{item['llm_explanation']}"
                            )
                        st.markdown("---")
                else:
                    data = response.json()

                    results = data.get("results", {})
                    percentage_results = results.get("percentage_results", [])
                    overall_avg = results.get("overall_avg", 0)

                    # Show result
                    st.subheader("Result")
                    st.markdown(
                        f"<span style='font-size:1.2rem; font-weight:normal;'> 📄 Average similarity percentage: {overall_avg}% </span>",
                        unsafe_allow_html=True
                    )

                    # Show matches per page
                    for page in percentage_results:
                        query_page = page.get("Query Page")
                        for attempt, match in enumerate(page.get("Matches", []), start=1):
                            matched_pdf = match.get("Matched PDF")
                            similarity = match.get("Similarity (%)")

                            # Extract PDF file path and page number
                            if "_page_" in matched_pdf:
                                pdf_file, page_num = matched_pdf.split("_page_", 1)
                            else:
                                pdf_file, page_num = matched_pdf, "N/A"

                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col1:
                                st.write(f"Input paper page {query_page}")
                                st.write(f"Attempt {attempt}")
                            with col2:
                                st.write(f"{pdf_file}")
                                st.write(f"Page {page_num}")
                            with col3:
                                st.write(f"🔍 Similarity: {similarity}%")

                            st.markdown("---")
                    

            else:
                st.error("Error: " + response.text)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the similarity service. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
