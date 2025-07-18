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

# Changed the model from 'gemini-pro' to 'gemini-2.0-flash' for broader availability and compatibility.
model = genai.GenerativeModel('gemini-2.0-flash')

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

def procedure_specific_validation(text, procedure_type):
    """
    AI-powered procedure-specific validation that knows exactly what each procedure type requires.
    """
    validation_prompt = f"""
    You are an expert in {procedure_type} documentation requirements. Based on the procedure type, validate if all required elements are present:

    PROCEDURE-SPECIFIC REQUIREMENTS DATABASE:

    **Excisional Debridement:**
    - Required: Anatomical location with laterality, depth of debridement (subcutaneous/fascia/muscle/bone), size measurements (length x width x depth), tissue type removed, instruments used, hemostasis method, wound appearance post-debridement
    - Critical Distinctions: "Excisional" vs "non-excisional" (affects MS-DRG 907-908 vs 573-574)

    **Appendectomy:**
    - Required: Approach (open/laparoscopic), appendix condition (perforated/gangrenous/simple), complications, specimen description
    - Critical Distinctions: Complicated vs uncomplicated appendicitis

    **Colonoscopy:**
    - Required: Extent of examination, quality of preparation, findings by segment, interventions performed, withdrawal time
    - Critical Distinctions: Screening vs diagnostic, polyp characteristics

    **Cardiac Catheterization:**
    - Required: Access site, vessels examined, findings per vessel, interventions, complications, closure method
    - Critical Distinctions: Diagnostic vs interventional

    **Skin Lesion Excision:**
    - Required: Lesion location, size, margins, closure type, specimen handling, pathology disposition
    - Critical Distinctions: Simple vs complex repair

    Analyze the provided text and identify:
    1. Missing required elements for this specific procedure type
    2. Elements that need more specificity
    3. Critical distinctions that affect coding/billing

    --- PROCEDURE TEXT ---
    {text}
    --- END TEXT ---

    Return JSON: {{"Missing_Elements": ["list"], "Needs_Specificity": ["list"], "Critical_Issues": ["list"], "Compliance_Score": int_0_to_10}}
    """
    
    try:
        response = model.generate_content(validation_prompt)
        json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        return json.loads(json_str)
    except:
        return {"Missing_Elements": [], "Needs_Specificity": [], "Critical_Issues": [], "Compliance_Score": 5}

def anatomical_intelligence_check(text):
    """
    AI that understands anatomy and flags imprecise locations.
    """
    anatomy_prompt = f"""
    You are an expert in anatomical precision for medical documentation. Analyze the text for anatomical location accuracy and specificity.

    ANATOMICAL PRECISION STANDARDS:
    
    **Specificity Levels (Best to Worst):**
    - Excellent: "Left lower extremity, anterior tibial region, distal third"
    - Good: "Left lower leg, anterior aspect"  
    - Fair: "Left lower leg"
    - Poor: "Lower extremity" or "Leg"
    - Unacceptable: "Extremity" or no location

    **Required Elements:**
    - Laterality (left/right/bilateral)
    - Anatomical region (specific body part)
    - Anatomical landmarks when relevant
    - Orientation (anterior/posterior/medial/lateral)
    - Level/section when relevant (proximal/middle/distal third)

    **Common Issues:**
    - Missing laterality
    - Vague terms ("area", "region" without specifics)
    - Inconsistent location descriptions
    - Missing anatomical landmarks for procedures

    Analyze the text and identify:
    1. All anatomical locations mentioned
    2. Precision level of each location
    3. Missing anatomical details
    4. Inconsistencies in location descriptions

    --- TEXT ---
    {text}
    --- END TEXT ---

    Return JSON: {{"Locations_Found": ["list with precision scores"], "Missing_Details": ["list"], "Inconsistencies": ["list"], "Overall_Precision": int_0_to_10}}
    """
    
    try:
        response = model.generate_content(anatomy_prompt)
        json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        return json.loads(json_str)
    except:
        return {"Locations_Found": [], "Missing_Details": [], "Inconsistencies": [], "Overall_Precision": 5}

def clinical_correlation_analysis(text):
    """
    AI that validates procedures match clinical indications.
    """
    correlation_prompt = f"""
    You are an expert in clinical correlation analysis. Validate that documented procedures logically match the clinical indications and findings.

    CLINICAL CORRELATION RULES:

    **Procedure-Indication Matching:**
    - Debridement → Necrotic tissue, infected wound, chronic ulcer
    - Appendectomy → Appendicitis, RLQ pain, elevated WBC
    - Colonoscopy → GI bleeding, screening, polyp surveillance
    - Cardiac cath → Chest pain, abnormal stress test, MI
    - Skin excision → Suspicious lesion, malignancy

    **Red Flags:**
    - Procedure without clear indication
    - Indication without supporting clinical evidence
    - Procedures that don't match severity of condition
    - Missing diagnostic workup before procedure

    **Clinical Evidence Requirements:**
    - Labs/imaging supporting diagnosis
    - Physical exam findings
    - Patient symptoms
    - Failed conservative measures (when applicable)

    Analyze the correlation between indications, clinical evidence, and procedures performed:

    --- TEXT ---
    {text}
    --- END TEXT ---

    Return JSON: {{"Indication_Match": int_0_to_10, "Missing_Evidence": ["list"], "Logic_Issues": ["list"], "Recommendations": ["list"]}}
    """
    
    try:
        response = model.generate_content(correlation_prompt)
        json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        return json.loads(json_str)
    except:
        return {"Indication_Match": 5, "Missing_Evidence": [], "Logic_Issues": [], "Recommendations": []}

def coding_prediction_analysis(text, procedure_type):
    """
    AI that predicts final codes and flags potential coding issues.
    """
    coding_prompt = f"""
    You are an expert in medical coding with deep knowledge of CPT, ICD-10-PCS, and MS-DRG assignments. Predict coding issues for this {procedure_type}.

    CODING PREDICTION DATABASE:

    **Debridement Procedures:**
    - CPT 11042-11047 (based on depth and size)
    - Critical: "Excisional" vs "Non-excisional" affects code selection
    - MS-DRG impact: 907-908 vs 573-574
    - Documentation needs: Depth, size, tissue type

    **Appendectomy:**
    - CPT 44970 (laparoscopic) vs 44960 (open)
    - ICD-10: K35-K37 (appendicitis variations)
    - MS-DRG 338-340 (complicated vs uncomplicated)

    **Skin Procedures:**
    - CPT based on size and complexity
    - Benign vs malignant affects DRG assignment
    - Repair complexity affects additional codes

    **Common Coding Issues:**
    - Insufficient detail for code specificity
    - Missing size measurements
    - Unclear procedure approach
    - Missing complication documentation
    - Inadequate anatomical specificity

    Based on the documentation, predict:
    1. Likely CPT codes
    2. Potential ICD-10 codes
    3. MS-DRG implications
    4. Documentation gaps that could affect coding
    5. Revenue impact of missing details

    --- TEXT ---
    {text}
    --- END TEXT ---

    Return JSON: {{"Predicted_CPT": ["list"], "Predicted_ICD10": ["list"], "MS_DRG_Risk": "str", "Coding_Gaps": ["list"], "Revenue_Impact": "str"}}
    """
    
    try:
        response = model.generate_content(coding_prompt)
        json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        return json.loads(json_str)
    except:
        return {"Predicted_CPT": [], "Predicted_ICD10": [], "MS_DRG_Risk": "Unknown", "Coding_Gaps": [], "Revenue_Impact": "Unknown"}

def generate_procedure_cdi_analysis(text, procedure_type="Unknown", retries=3):
    """
    Enhanced CDI analysis with AI-powered procedure-specific validation, anatomical intelligence,
    clinical correlation, and coding prediction.
    """
    
    # Run the 4 advanced AI analyses
    procedure_validation = procedure_specific_validation(text, procedure_type)
    anatomical_analysis = anatomical_intelligence_check(text)
    clinical_correlation = clinical_correlation_analysis(text)
    coding_prediction = coding_prediction_analysis(text, procedure_type)
    
    procedure_cdi_criteria = """
    You are a Senior CDI Specialist with advanced AI-powered analytical capabilities. Using the provided AI analysis results, evaluate this procedure document against the 5 critical criteria:

    1. **Clinical Reliability (Score 0-10):**
        - Do the procedures performed match the documented indications?
        - Is the preoperative diagnosis supported by the procedure findings?
        - Are treatments/interventions consistent with documented conditions?
        
    2. **Clinical Precision (Score 0-10):**
        - Are anatomical locations specific and accurate?
        - Is procedure type clearly specified with all critical distinctions?
        - Are measurements exact and complete?
        - Are instruments and techniques precisely documented?
        
    3. **Documentation Completeness (Score 0-10):**
        - All required elements present for this specific procedure type?
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

    For each criterion with score <10, provide a specific, compliant CDI query that addresses the deficiency.
    """

    prompt = f"""
    {procedure_cdi_criteria}

    ADVANCED AI ANALYSIS RESULTS:
    
    **Procedure-Specific Validation:**
    - Missing Elements: {procedure_validation.get('Missing_Elements', [])}
    - Needs Specificity: {procedure_validation.get('Needs_Specificity', [])}
    - Critical Issues: {procedure_validation.get('Critical_Issues', [])}
    - Compliance Score: {procedure_validation.get('Compliance_Score', 5)}/10
    
    **Anatomical Intelligence:**
    - Locations Found: {anatomical_analysis.get('Locations_Found', [])}
    - Missing Details: {anatomical_analysis.get('Missing_Details', [])}
    - Inconsistencies: {anatomical_analysis.get('Inconsistencies', [])}
    - Precision Score: {anatomical_analysis.get('Overall_Precision', 5)}/10
    
    **Clinical Correlation:**
    - Indication Match Score: {clinical_correlation.get('Indication_Match', 5)}/10
    - Missing Evidence: {clinical_correlation.get('Missing_Evidence', [])}
    - Logic Issues: {clinical_correlation.get('Logic_Issues', [])}
    - Recommendations: {clinical_correlation.get('Recommendations', [])}
    
    **Coding Prediction:**
    - Predicted CPT: {coding_prediction.get('Predicted_CPT', [])}
    - Predicted ICD-10: {coding_prediction.get('Predicted_ICD10', [])}
    - MS-DRG Risk: {coding_prediction.get('MS_DRG_Risk', 'Unknown')}
    - Coding Gaps: {coding_prediction.get('Coding_Gaps', [])}
    - Revenue Impact: {coding_prediction.get('Revenue_Impact', 'Unknown')}

    Using these AI insights, provide comprehensive CDI analysis in this JSON format:

    {{
      "Procedure_Analysis": {{
        "Clinical_Reliability": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Clinical_Precision": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Documentation_Completeness": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Clinical_Consistency": {{"Score": int, "Issues": "str", "CDI_Query": "str"}},
        "Clarity": {{"Score": int, "Issues": "str", "CDI_Query": "str"}}
      }},
      "AI_Enhanced_Findings": {{
        "Procedure_Validation_Issues": {procedure_validation.get('Critical_Issues', [])},
        "Anatomical_Precision_Issues": {anatomical_analysis.get('Missing_Details', [])},
        "Clinical_Correlation_Issues": {clinical_correlation.get('Logic_Issues', [])},
        "Coding_Risk_Factors": {coding_prediction.get('Coding_Gaps', [])}
      }},
      "Critical_Findings": {{
        "High_Priority_Issues": ["AI-identified most critical documentation deficiencies"],
        "MS_DRG_Impact": "{coding_prediction.get('MS_DRG_Risk', 'Unknown')}",
        "Revenue_Impact": "{coding_prediction.get('Revenue_Impact', 'Unknown')}",
        "Coding_Clarifications_Needed": {coding_prediction.get('Coding_Gaps', [])}
      }},
      "Overall_Assessment": {{
        "Total_Score": "sum of all scores out of 50",
        "Quality_Grade": "A/B/C/D/F based on total score",
        "Summary": "AI-enhanced assessment focusing on procedure-specific issues",
        "AI_Confidence": "High/Medium/Low based on analysis quality"
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
            result = json.loads(json_str)
            
            # Add the individual AI analysis results to the final output
            result["Detailed_AI_Analysis"] = {
                "Procedure_Validation": procedure_validation,
                "Anatomical_Analysis": anatomical_analysis,
                "Clinical_Correlation": clinical_correlation,
                "Coding_Prediction": coding_prediction
            }
            
            return result
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
                
                # Perform AI-enhanced procedure analysis
                analysis_result = generate_procedure_cdi_analysis(
                    section_text,  
                    proc_info.get("Procedure Type", "Unknown")
                )
                
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
                
                # Display AI-enhanced insights
                # The 'result' variable here is from the last iteration of the loop, which might not be what's intended.
                # It's better to iterate through all_results to display details for each section or pick a representative one.
                # For simplicity, I'll display the details for the first section if available.
                first_section_result = next(iter(all_results.values()), None)
                if first_section_result and 'Detailed_AI_Analysis' in first_section_result:
                    with st.expander("🤖 AI-Enhanced Analysis Details (for first analyzed section)"):
                        ai_analysis = first_section_result['Detailed_AI_Analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🔍 Procedure Validation")
                            pv = ai_analysis['Procedure_Validation']
                            st.metric("Compliance Score", f"{pv.get('Compliance_Score', 0)}/10")
                            if pv.get('Missing_Elements'):
                                st.error("Missing Elements:")
                                for elem in pv['Missing_Elements']:
                                    st.write(f"• {elem}")
                            
                            st.subheader("🎯 Coding Prediction")
                            cp = ai_analysis['Coding_Prediction']
                            if cp.get('Predicted_CPT'):
                                st.info(f"Predicted CPT: {', '.join(cp['Predicted_CPT'])}")
                            if cp.get('MS_DRG_Risk'):
                                st.warning(f"MS-DRG Risk: {cp['MS_DRG_Risk']}")
                        
                        with col2:
                            st.subheader("📍 Anatomical Intelligence")
                            aa = ai_analysis['Anatomical_Analysis']
                            st.metric("Precision Score", f"{aa.get('Overall_Precision', 0)}/10")
                            if aa.get('Missing_Details'):
                                st.warning("Needs More Anatomical Detail:")
                                for detail in aa['Missing_Details']:
                                    st.write(f"• {detail}")
                            
                            st.subheader("🔗 Clinical Correlation")
                            cc = ai_analysis['Clinical_Correlation']
                            st.metric("Correlation Score", f"{cc.get('Indication_Match', 0)}/10")
                            if cc.get('Logic_Issues'):
                                st.error("Logic Issues:")
                                for issue in cc['Logic_Issues']:
                                    st.write(f"• {issue}")
                
                # Enhanced summary metrics with AI insights
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
                    for section_name, section_result in all_results.items():
                        if 'Overall_Assessment' in section_result:
                            total_score = section_result['Overall_Assessment'].get('Total_Score', '0/50')
                            if isinstance(total_score, str) and '/' in total_score:
                                score = int(total_score.split('/')[0])
                                avg_scores.append(score)
                    avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0
                    st.metric("Avg Quality Score", f"{avg_score:.1f}/50")
                with col4:
                    # AI confidence scoring
                    ai_confidence_scores = []
                    for section_result in all_results.values():
                        if 'Overall_Assessment' in section_result:
                            ai_conf = section_result['Overall_Assessment'].get('AI_Confidence', 'Medium')
                            if ai_conf == 'High':
                                ai_confidence_scores.append(3)
                            elif ai_conf == 'Medium':
                                ai_confidence_scores.append(2)
                            else:
                                ai_confidence_scores.append(1)
                    avg_confidence = sum(ai_confidence_scores) / len(ai_confidence_scores) if ai_confidence_scores else 2
                    confidence_label = "High" if avg_confidence >= 2.5 else "Medium" if avg_confidence >= 1.5 else "Low"
                    st.metric("AI Confidence", confidence_label)

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
                    
                    for section_name, section_result in all_results.items():
                        summary_report += f"SECTION: {section_name.title()}\n"
                        summary_report += f"Document Type: {section_result.get('Doc Type', 'Unknown')}\n"
                        summary_report += f"Procedure Type: {section_result.get('Procedure Type', 'Unknown')}\n"
                        
                        if 'Overall_Assessment' in section_result:
                            oa = section_result['Overall_Assessment']
                            summary_report += f"Overall Score: {oa.get('Total_Score', 'N/A')}\n"
                            summary_report += f"Quality Grade: {oa.get('Quality_Grade', 'N/A')}\n"
                            summary_report += f"Summary: {oa.get('Summary', 'N/A')}\n"
                        
                        if 'AI_Enhanced_Findings' in section_result:
                            aef = section_result['AI_Enhanced_Findings']
                            summary_report += f"\n--- AI-ENHANCED FINDINGS ---\n"
                            if aef.get('Procedure_Validation_Issues'):
                                summary_report += f"Procedure Validation Issues: {'; '.join(aef['Procedure_Validation_Issues'])}\n"
                            if aef.get('Anatomical_Precision_Issues'):
                                summary_report += f"Anatomical Precision Issues: {'; '.join(aef['Anatomical_Precision_Issues'])}\n"
                            if aef.get('Clinical_Correlation_Issues'):
                                summary_report += f"Clinical Correlation Issues: {'; '.join(aef['Clinical_Correlation_Issues'])}\n"
                            if aef.get('Coding_Risk_Factors'):
                                summary_report += f"Coding Risk Factors: {'; '.join(aef['Coding_Risk_Factors'])}\n"
                        
                        if 'Detailed_AI_Analysis' in section_result:
                            da = section_result['Detailed_AI_Analysis']
                            summary_report += f"\n--- DETAILED AI INSIGHTS ---\n"
                            if 'Coding_Prediction' in da:
                                cp = da['Coding_Prediction']
                                summary_report += f"Predicted CPT Codes: {', '.join(cp.get('Predicted_CPT', []))}\n"
                                summary_report += f"Revenue Impact: {cp.get('Revenue_Impact', 'Unknown')}\n"
                        
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
