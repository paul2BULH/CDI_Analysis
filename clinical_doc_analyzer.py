# Streamlit app: Inpatient CDI Analyzer
import streamlit as st
import fitz  # PyMuPDF
import docx
import tempfile
import os
import pandas as pd
from google.generativeai import GenerativeModel
import google.generativeai as genai
import json
import time
from io import BytesIO

# --- Load Gemini API Key ---
if 'gemini' in st.secrets:
    genai.configure(api_key=st.secrets['gemini']['api_key'])
else:
    st.error("❌ Gemini API key not configured. Please set it in .streamlit/secrets.toml")
    st.stop()

model = genai.GenerativeModel('gemini-pro')

# --- Helper Functions ---
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    return text

def extract_text_from_docx(file):
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())

def split_sections(text):
    headers = ["history of present illness", "progress note", "operative report", "discharge summary", "consultation", "physical exam", "assessment and plan"]
    lines = text.lower().splitlines()
    sections = {}
    current_section = "unknown"
    buffer = []
    for line in lines:
        if any(h in line for h in headers):
            if buffer:
                sections[current_section] = "\n".join(buffer).strip()
                buffer = []
            current_section = line.strip()
        else:
            buffer.append(line)
    if buffer:
        sections[current_section] = "\n".join(buffer).strip()
    return sections

def categorize_section_type(text):
    snippet = text[:1000]
    prompt = f"""
What type of clinical documentation is this?
Choose one: H&P, Progress Note, Operative Report, Discharge Summary, Consultation, Other.
Provide confidence %.

--- Text ---
{snippet}
--- End ---
Respond in JSON like: {{"Document Type": str, "Confidence": int}}
"""
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except:
        return {{"Document Type": "Unknown", "Confidence": 0}}

def generate_gemini_analysis(text, retries=2):
    prompt = f"""
You are a Senior Clinical Documentation Integrity (CDI) Specialist. Assess the following clinical document using the theory of high-quality documentation:
1. Clinical Reliability
2. Clinical Precision
3. Documentation Completeness
4. Clarity and Consistency
Also flag timeliness issues.

For each criterion:
- Assign a score (0-10)
- Explain why the score was given
- Suggest a CDI Action
- If score < 10, draft a sample compliant physician query

Return your output in structured JSON like this:
{{
  "Scorecard": {{
    "Clinical Reliability": {{"Score": int, "Why": str, "CDI Action": str, "Draft Query": str}},
    ...
  }},
  "Timeliness Flag": str,
  "Final Summary": str
}}

--- Document Start ---
{text}
--- Document End ---
"""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            time.sleep(2)
    return {{"error": "Gemini failed after retries."}}

def generate_excel(results):
    df_rows = []
    for section, result in results.items():
        scorecard = result.get("Scorecard", {{}})
        doc_type = result.get("Doc Type", "")
        confidence = result.get("Confidence", "")
        for criterion, detail in scorecard.items():
            score = detail.get("Score", "")
            df_rows.append({{
                "Section": section.title(),
                "AI Doc Type": doc_type,
                "Confidence %": confidence,
                "Criterion": criterion,
                "Score": score,
                "Why": detail.get("Why", ""),
                "CDI Action": detail.get("CDI Action", ""),
                "Draft Query": detail.get("Draft Query", "")
            }})
    return pd.DataFrame(df_rows)

def highlight_score(val):
    try:
        score = int(val)
        if score <= 6:
            return 'background-color: #ffcccc'  # red
        elif score <= 8:
            return 'background-color: #fff2cc'  # yellow
        else:
            return 'background-color: #d9ead3'  # green
    except:
        return ''

# --- Streamlit UI ---
st.set_page_config(page_title="Inpatient CDI Analyzer", layout="wide")
st.title("🩺 Inpatient CDI Analyzer")

uploaded_file = st.file_uploader("📤 Upload Full Inpatient Chart (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        full_text = extract_text_from_pdf(uploaded_file)
    else:
        full_text = extract_text_from_docx(uploaded_file)

    st.success("✅ Document Uploaded and Processed")

    sections = split_sections(full_text)
    st.subheader("📑 Detected Sections")

    all_results = {{}}

    if st.button("🧠 Analyze All Sections"):
        for i, (section_name, section_text) in enumerate(sections.items()):
            with st.spinner(f"Analyzing: {section_name.title()}..."):
                doc_info = categorize_section_type(section_text)
                result = generate_gemini_analysis(section_text)
                if 'error' not in result and result.get("Scorecard"):
                    result["Doc Type"] = doc_info.get("Document Type")
                    result["Confidence"] = doc_info.get("Confidence")
                    all_results[section_name] = result
                else:
                    st.warning(f"❌ Failed to analyze section: {section_name.title()}")

        st.success("✅ All sections analyzed!")

        df = generate_excel(all_results)
        if 'Score' in df.columns:
            styled_df = df.style.applymap(highlight_score, subset=['Score'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("⚠️ No valid 'Score' column found in results. Showing raw output.")
            st.dataframe(df, use_container_width=True)

        with BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel", buffer.getvalue(), file_name="CDI_Analysis.xlsx")

        txt_summary = ""
        for sec, res in all_results.items():
            txt_summary += f"\n\n=== {sec.title()} ({res.get('Doc Type', '')}, Confidence: {res.get('Confidence', '')}%) ===\n"
            if 'error' in res:
                txt_summary += "Error during analysis."
            else:
                for k, v in res.get("Scorecard", {{}}).items():
                    txt_summary += f"\n{k} ({v.get('Score', '-')}/10): {v.get('Why', '')}"
                    if int(v.get("Score", 10)) < 10:
                        txt_summary += f"\n→ Draft Query: {v.get('Draft Query', '')}"
        st.download_button("📄 Download TXT Report", txt_summary, file_name="CDI_Queries_Report.txt")

    for i, (section_name, section_text) in enumerate(sections.items()):
        with st.expander(f"📄 Section {i+1}: {section_name.title()}"):
            st.text(section_text[:2000])
            if st.button(f"🔍 Analyze '{section_name.title()}'", key=f"analyze_{i}"):
                with st.spinner("Analyzing with Gemini..."):
                    doc_info = categorize_section_type(section_text)
                    result = generate_gemini_analysis(section_text)
                    result["Doc Type"] = doc_info.get("Document Type")
                    result["Confidence"] = doc_info.get("Confidence")
                    st.json(result)
