# Streamlit app: Procedure-Focused CDI Analyzer
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
    """Extracts text from a PDF file using PyMuPDF."""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    return text

def extract_text_from_docx(file):
    """Extracts text from a DOCX file using python-docx."""
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())

def split_procedure_sections(text):
    """
    Splits procedure documentation into logical sections based on common procedure headers.
    """
    procedure_headers = [
        "operative report", "procedure report", "surgery report",
        "preoperative diagnosis", "postoperative diagnosis", "final diagnosis",
        "procedure performed", "operation performed", "surgical procedure",
        "indications", "indications for procedure", "indication for surgery",
        "operative technique", "procedure technique", "surgical technique",
        "operative findings", "intraoperative findings", "surgical findings",
        "complications", "intraoperative complications", "postoperative complications",
        "specimens", "pathology", "tissue removed", "specimens sent",
        "anesthesia", "anesthesia record", "anesthesia type",
        "preoperative note", "postoperative note", "recovery note",
        "wound details", "closure", "dressing", "hemostasis",
        "estimated blood loss", "fluids", "positioning",
        "consent", "informed consent"
    ]
    
    lines = text.lower().splitlines()
    sections = {}
    current_section = "procedure_overview"  # Default section
    buffer = []

    for line in lines:
        found_header = False
        for h in procedure_headers:
            # More flexible header matching for procedures
            if h in line and len(line) < 150:  # Allow slightly longer lines for procedure headers
                if buffer:
                    sections[current_section.strip()] = "\n".join(buffer).strip()
                    buffer = []
                current_section = line.strip()
                found_header = True
                break
        if not found_header:
            buffer.append(line)

    # Add any remaining text
    if buffer:
        sections[current_section.strip()] = "\n".join(buffer).strip()
    
    return sections

def categorize_procedure_type(text):
    """
    Uses Gemini to categorize the specific type of procedure documentation.
    """
    snippet = text[:2000]  # Larger snippet for procedures
    prompt = f"""
    You are an expert in surgical and procedure documentation. Analyze this text to determine the specific type of procedure document.
    
    Choose the most specific category from:
    - Operative Report (major surgery)
    - Procedure Report (minor procedure)
    - Endoscopic Report
    - Interventional Radiology Report
    - Cardiac Catheterization Report
    - Anesthesia Record
    - Pre-operative Assessment
    - Post-operative Note
    - Wound Care Procedure Note
    - Debridement Procedure (Excisional/Non-excisional)
    - Biopsy Report
    - Other/Unknown
    
    Also identify the main procedure type (e.g., "Excisional Debridement", "Appendectomy", "Endoscopy").
    Provide confidence percentage.

    --- Text Snippet ---
    {snippet}
    --- End Snippet ---
    
    Respond in JSON format: {{"Document Type": "str", "Procedure Type": "str", "Confidence": int}}
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        return json.loads(json_str)
    except Exception as e:
        st.warning(f"Error categorizing procedure type: {e}")
        return {"Document Type": "Unknown", "Procedure Type": "Unknown", "Confidence": 0}

def generate_procedure_cdi_analysis(text, retries=3):
    """
    Specialized CDI analysis for procedure documentation focusing on the 7 critical criteria
    with procedure-specific emphasis.
    """
    
    procedure_cdi_criteria = """
    You are a Senior CDI Specialist specializing in PROCEDURE DOCUMENTATION. Evaluate this procedure document against these 5 critical criteria:

    1. **Clinical Reliability (Score 0-10):**
       - Do the procedures performed match the documented indications?
       - Is the preoperative diagnosis supported by the procedure findings?
       - Are treatments/interventions consistent with documented conditions?
       - Example Issue: Excisional debridement performed but no clear indication documented
       
    2. **Clinical Precision (Score 0-10):**
       - Are anatomical locations specific? (e.g., "left lower leg" vs "leg")
       - Is procedure type clearly specified? (e.g., "excisional" vs "non-excisional" debridement)
       - Are measurements exact and complete?
       - Are instruments and techniques precisely documented?
       
    3. **Documentation Completeness (Score 0-10):**
       - All required elements present: indications, technique, findings, specimens, complications?
       - Are abnormal findings addressed with clinical significance?
       - Missing elements that should be documented?
       
    4. **Clinical Consistency (Score 0-10):**
       - Do preoperative and postoperative diagnoses align logically?
       - Are procedure findings consistent with documented technique?
       - Any contradictions between different sections of the report?
       
    5. **Clarity (Score 0-10):**
       - Is the procedure description unambiguous?
       - Are findings clearly explained?
       - Would another surgeon understand exactly what was done?
       - Any vague terms that need clarification?

    For each criterion with score <10, provide a specific, compliant CDI query that addresses the deficiency.
    """

    prompt = f"""
    {procedure_cdi_criteria}

    PROCEDURE-SPECIFIC FOCUS AREAS:
    - Excisional vs Non-excisional debridement distinction (critical for MS-DRG assignment)
    - Specific anatomical locations and laterality
    - Depth of tissue involvement (subcutaneous, fascia, muscle, bone)
    - Wound measurements and characteristics
    - Instruments used and technique details
    - Specimens removed and pathology correlation
    - Complications or absence thereof

    Analyze the following procedure documentation and return results in this JSON format:

    {{
      "Procedure_Analysis": {{
        "Clinical_Reliability": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Clinical_Precision": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Documentation_Completeness": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Clinical_Consistency": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Clarity": {{"Score": int, "Issues": "str", "CDI_Query": "str"}}
      }},
      "Critical_Findings": {{
        "High_Priority_Issues": ["list of most critical documentation deficiencies"],
        "MS_DRG_Impact": "potential impact on DRG assignment",
        "Coding_Clarifications_Needed": ["specific areas needing physician clarification"]
      }},
      "Overall_Assessment": {{
        "Total_Score": "sum of all scores out of 50",
        "Quality_Grade": "A/B/C/D/F based on total score",
        "Summary": "concise overall assessment focusing on procedure-specific issues"
      }}
    }}

    --- PROCEDURE DOCUMENT ---
    {text}
    --- END DOCUMENT ---
    """
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            st.warning(f"Attempt {attempt+1}: JSON decoding error. Retrying...")
            time.sleep(2)
        except Exception as e:
            st.warning(f"Attempt {attempt+1}: Analysis error: {e}. Retrying...")
            time.sleep(2)
    
    return {"error": "Analysis failed after retries", "raw_response": response.text if 'response' in locals() else "No response"}

def generate_procedure_excel_report(results):
    """
    Generates a specialized Excel report for procedure CDI analysis.
    """
    df_rows = []
    
    for section, result in results.items():
        if 'error' in result:
            continue
            
        doc_type = result.get("Doc Type", "")
        procedure_type = result.get("Procedure Type", "")
        confidence = result.get("Confidence", "")
        
        procedure_analysis = result.get("Procedure_Analysis", {})
        critical_findings = result.get("Critical_Findings", {})
        overall_assessment = result.get("Overall_Assessment", {})
        
        # Overall section summary
        df_rows.append({
            "Section": section.title(),
            "Document_Type": doc_type,
            "Procedure_Type": procedure_type,
            "Confidence_%": confidence,
            "Criterion": "OVERALL ASSESSMENT",
            "Score": overall_assessment.get("Total_Score", ""),
            "Grade": overall_assessment.get("Quality_Grade", ""),
            "Issues_Identified": overall_assessment.get("Summary", ""),
            "CDI_Query": "",
            "Priority": "Summary"
        })
        
        # High priority issues
        high_priority = critical_findings.get("High_Priority_Issues", [])
        if high_priority:
            df_rows.append({
                "Section": section.title(),
                "Document_Type": doc_type,
                "Procedure_Type": procedure_type,
                "Confidence_%": confidence,
                "Criterion": "HIGH PRIORITY ISSUES",
                "Score": "",
                "Grade": "",
                "Issues_Identified": "; ".join(high_priority),
                "CDI_Query": "",
                "Priority": "HIGH"
            })
        
        # Individual criteria
        for criterion, details in procedure_analysis.items():
            score = details.get("Score", 0)
            issues = details.get("Issues", "")
            query = details.get("CDI_Query", "")
            
            # Determine priority based on score
            if score <= 5:
                priority = "HIGH"
            elif score <= 7:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            # Only include query if score is less than 10
            display_query = query if score < 10 else ""
            
            df_rows.append({
                "Section": section.title(),
                "Document_Type": doc_type,
                "Procedure_Type": procedure_type,
                "Confidence_%": confidence,
                "Criterion": criterion.replace("_", " ").title(),
                "Score": f"{score}/10",
                "Grade": "",
                "Issues_Identified": issues,
                "CDI_Query": display_query,
                "Priority": priority
            })
    
    return pd.DataFrame(df_rows)

def highlight_procedure_scores(val):
    """Enhanced highlighting for procedure scores with priority colors."""
    if isinstance(val, str) and "/" in val:
        try:
            score = int(val.split("/")[0])
            if score >= 9:
                return 'background-color: #d9ead3; font-weight: bold'  # Green
            elif score >= 7:
                return 'background-color: #fff2cc; font-weight: bold'  # Yellow
            elif score >= 5:
                return 'background-color: #fce5cd; font-weight: bold'  # Orange
            else:
                return 'background-color: #f4cccc; font-weight: bold'  # Red
        except:
            return ''
    return ''

def highlight_priority(val):
    """Color code by priority level."""
    if val == "HIGH":
        return 'background-color: #f4cccc; color: #cc0000; font-weight: bold'
    elif val == "MEDIUM":
        return 'background-color: #fce5cd; color: #e69138; font-weight: bold'
    elif val == "LOW":
        return 'background-color: #d9ead3; color: #6aa84f; font-weight: bold'
    return ''

# --- Streamlit UI ---
st.set_page_config(page_title="Procedure CDI Analyzer", layout="wide")
st.title("🏥 Procedure-Focused CDI Analyzer")
st.markdown("**Specialized for Operative Reports, Procedure Notes, and Surgical Documentation**")

# Sidebar with procedure-specific guidance
with st.sidebar:
    st.header("🔍 Procedure Documentation Focus")
    st.markdown("""
    **This analyzer specializes in:**
    - Operative Reports
    - Procedure Notes  
    - Surgical Documentation
    - Debridement Procedures
    - Anesthesia Records
    
    **Key Areas Evaluated:**
    - Excisional vs Non-excisional specification
    - Anatomical precision & laterality
    - Procedure indications & findings
    - Technique documentation
    - Specimen/pathology correlation
    """)

uploaded_file = st.file_uploader("📤 Upload Procedure Document (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file:
    with st.spinner("Processing procedure document..."):
        if uploaded_file.name.endswith(".pdf"):
            full_text = extract_text_from_pdf(uploaded_file)
        else:
            full_text = extract_text_from_docx(uploaded_file)

    st.success("✅ Procedure Document Processed!")

    # Quick document preview
    with st.expander("📄 Document Preview (First 1000 characters)"):
        st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)

    sections = split_procedure_sections(full_text)
    st.subheader(f"📑 Detected Procedure Sections ({len(sections)} found)")

    # Display sections found
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sections Identified:**")
        for i, section_name in enumerate(sections.keys(), 1):
            st.write(f"{i}. {section_name.title()}")

    all_results = {}

    # Analyze button
    if st.button("🧠 Analyze Procedure Documentation", type="primary"):
        progress_bar = st.progress(0)
        total_sections = len(sections)
        
        for i, (section_name, section_text) in enumerate(sections.items()):
            st.info(f"Analyzing: **{section_name.title()}** ({i+1}/{total_sections})")
            
            with st.spinner(f"CDI Analysis in progress..."):
                # Categorize procedure type
                proc_info = categorize_procedure_type(section_text)
                
                # Perform specialized procedure analysis
                analysis_result = generate_procedure_cdi_analysis(section_text)
                
                if 'error' not in analysis_result:
                    analysis_result["Doc Type"] = proc_info.get("Document Type")
                    analysis_result["Procedure Type"] = proc_info.get("Procedure Type")
                    analysis_result["Confidence"] = proc_info.get("Confidence")
                    all_results[section_name] = analysis_result
                    st.success(f"✅ **{section_name.title()}** analyzed successfully")
                else:
                    st.error(f"❌ Failed to analyze **{section_name.title()}**")
                    st.code(analysis_result.get('raw_response', 'No response'), language='text')
            
            progress_bar.progress((i + 1) / total_sections)
        
        st.success("🎉 Procedure Analysis Complete!")

        # Display comprehensive results
        if all_results:
            st.subheader("📊 Procedure CDI Analysis Results")
            
            df = generate_procedure_excel_report(all_results)
            
            if not df.empty:
                # Apply styling
                styled_df = df.style.applymap(highlight_procedure_scores, subset=['Score']) \
                                 .applymap(highlight_priority, subset=['Priority'])
                
                st.dataframe(styled_df, use_container_width=True, height=600)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate summary stats
                high_priority_count = len(df[df['Priority'] == 'HIGH'])
                medium_priority_count = len(df[df['Priority'] == 'MEDIUM'])
                
                with col1:
                    st.metric("High Priority Issues", high_priority_count)
                with col2:
                    st.metric("Medium Priority Issues", medium_priority_count)
                with col3:
                    avg_scores = []
                    for section, result in all_results.items():
                        if 'Overall_Assessment' in result:
                            total_score = result['Overall_Assessment'].get('Total_Score', '0/50')
                            if isinstance(total_score, str) and '/' in total_score:
                                score = int(total_score.split('/')[0])
                                avg_scores.append(score)
                    avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0
                    st.metric("Avg Quality Score", f"{avg_score:.1f}/50")
                with col4:
                    procedure_types = set()
                    for result in all_results.values():
                        if result.get('Procedure Type'):
                            procedure_types.add(result['Procedure Type'])
                    st.metric("Procedure Types", len(procedure_types))

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    # Excel download
                    with BytesIO() as buffer:
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Procedure_CDI_Analysis')
                            
                            # Get workbook and worksheet
                            workbook = writer.book
                            worksheet = writer.sheets['Procedure_CDI_Analysis']
                            
                            # Add formatting
                            high_priority_format = workbook.add_format({
                                'bg_color': '#f4cccc',
                                'font_color': '#cc0000',
                                'bold': True
                            })
                            
                            # Apply conditional formatting
                            worksheet.conditional_format('I:I', {
                                'type': 'text',
                                'criteria': 'containing',
                                'value': 'HIGH',
                                'format': high_priority_format
                            })
                        
                        st.download_button(
                            "📥 Download Excel Report", 
                            buffer.getvalue(), 
                            file_name="Procedure_CDI_Analysis.xlsx", 
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                with col2:
                    # Generate summary report
                    summary_report = "PROCEDURE CDI ANALYSIS SUMMARY REPORT\n"
                    summary_report += "=" * 50 + "\n\n"
                    
                    for section, result in all_results.items():
                        summary_report += f"SECTION: {section.title()}\n"
                        summary_report += f"Document Type: {result.get('Doc Type', 'Unknown')}\n"
                        summary_report += f"Procedure Type: {result.get('Procedure Type', 'Unknown')}\n"
                        
                        if 'Overall_Assessment' in result:
                            oa = result['Overall_Assessment']
                            summary_report += f"Overall Score: {oa.get('Total_Score', 'N/A')}\n"
                            summary_report += f"Quality Grade: {oa.get('Quality_Grade', 'N/A')}\n"
                            summary_report += f"Summary: {oa.get('Summary', 'N/A')}\n"
                        
                        if 'Critical_Findings' in result:
                            cf = result['Critical_Findings']
                            high_priority = cf.get('High_Priority_Issues', [])
                            if high_priority:
                                summary_report += f"\nHIGH PRIORITY ISSUES:\n"
                                for issue in high_priority:
                                    summary_report += f"• {issue}\n"
                            
                            ms_drg = cf.get('MS_DRG_Impact', '')
                            if ms_drg:
                                summary_report += f"MS-DRG Impact: {ms_drg}\n"
                        
                        summary_report += "\n" + "-" * 40 + "\n\n"
                    
                    st.download_button(
                        "📄 Download Summary Report", 
                        summary_report, 
                        file_name="Procedure_CDI_Summary.txt", 
                        mime="text/plain"
                    )

# Information section
st.markdown("---")
st.subheader("ℹ️ About Procedure CDI Analysis")
st.markdown("""
This specialized analyzer focuses on the unique documentation requirements for procedures and operative reports:

**Key Evaluation Areas:**
- **Precision**: Specific procedure types, anatomical locations, measurements
- **Completeness**: All required elements (indications, technique, findings, complications)
- **Reliability**: Procedures match documented indications and findings
- **MS-DRG Impact**: Critical distinctions affecting reimbursement (e.g., excisional vs non-excisional debridement)

**Common Procedure Documentation Issues:**
- Vague procedure descriptions
- Missing anatomical specificity
- Incomplete wound/specimen documentation
- Inconsistent pre/post-operative diagnoses
- Missing complication documentation
""")
