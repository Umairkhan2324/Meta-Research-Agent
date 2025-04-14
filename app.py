import streamlit as st
import logging
import os
import json
from main import ResearchState, workflow
import matplotlib.pyplot as plt

# Mark that we're running in Streamlit
os.environ["STREAMLIT_RUNNING"] = "1"

# Configure page
st.set_page_config(page_title="Meta-Analysis Assistant", layout="wide")

# Setup logging to display in Streamlit
logging.basicConfig(level=logging.INFO)

def main():
    st.title("Neurosurgery Meta-Analysis Assistant")
    
    # Initialize session state
    if 'state' not in st.session_state:
        st.session_state.state = {
            "topic": "",
            "pico": {},
            "articles": [],
            "included_articles": [],
            "extracted_data": [],
            "meta_results": {},
            "report_content": "",
            "human_approval": "",
            "stage": "start",
            "errors": []
        }
    
    # For tracking workflow advancement
    if 'next_stage' not in st.session_state:
        st.session_state.next_stage = None
    
    # Sidebar for workflow stages
    st.sidebar.title("Workflow Status")
    st.sidebar.write(f"Current Stage: {st.session_state.state['stage']}")
    
    # Add a recovery panel in the Streamlit sidebar
    # Insert this after the sidebar workflow status
    st.sidebar.write("---")
    if st.sidebar.checkbox("Show Advanced Options"):
        st.sidebar.subheader("Workflow Recovery")
        
        # Option to restart from a specific stage
        recovery_stages = ["start", "topic_received", "pico_defined", "search_completed", 
                          "screen_completed", "extraction_completed"]
        
        recovery_stage = st.sidebar.selectbox(
            "Restart from stage:", 
            recovery_stages, 
            disabled=st.session_state.state['stage'] == 'start'
        )
        
        if st.sidebar.button("Reset to Selected Stage"):
            # Save important data
            topic = st.session_state.state.get('topic', '')
            pico = st.session_state.state.get('pico', {})
            
            # Reset state to the selected stage
            st.session_state.state = {
                "topic": topic,
                "pico": pico if recovery_stage != 'start' else {},
                "articles": [] if recovery_stage in ['start', 'topic_received', 'pico_defined'] else st.session_state.state.get('articles', []),
                "included_articles": [] if recovery_stage in ['start', 'topic_received', 'pico_defined', 'search_completed'] else st.session_state.state.get('included_articles', []),
                "extracted_data": [] if recovery_stage != 'extraction_completed' else st.session_state.state.get('extracted_data', []),
                "meta_results": {},
                "report_content": "",
                "human_approval": "",
                "stage": recovery_stage,
                "errors": []
            }
            st.sidebar.success(f"Workflow reset to {recovery_stage}")
            st.rerun()

        # Option to clear all error messages
        if st.session_state.state.get('errors') and len(st.session_state.state['errors']) > 0:
            if st.sidebar.button("Clear Error Messages"):
                st.session_state.state['errors'] = []
                st.sidebar.success("Errors cleared")
                st.rerun()
    
    # Topic Input
    if st.session_state.state['stage'] in ['start', 'topic_received']:
        st.header("Research Topic")
        topic = st.text_area(
            "Enter your neurosurgery research topic:",
            value=st.session_state.state['topic'],
            help="e.g., 'Comparison of outcomes between minimally invasive and open craniotomy for tumor resection'"
        )
        if st.button("Submit Topic") and len(topic) > 10:
            with st.spinner("Processing topic..."):
                st.session_state.state['topic'] = topic
                
                # Rebuild the graph to ensure proper initialization
                from main import workflow
                graph = workflow.compile()
                
                try:
                    # Run the first part of the workflow
                    current_state = st.session_state.state.copy()
                    
                    # Process each step manually to avoid execution errors
                    from main import get_user_topic, define_pico, search_literature, screen_articles, extract_and_assess
                    
                    # Step by step manual execution to avoid graph errors
                    current_state = get_user_topic(current_state)
                    current_state = define_pico(current_state)
                    current_state = search_literature(current_state)
                    current_state = screen_articles(current_state)
                    current_state = extract_and_assess(current_state)
                    
                    # Set the stage to extraction_completed
                    current_state["stage"] = "extraction_completed"
                    st.session_state.state = current_state
                    st.rerun()
                except Exception as e:
                    st.error(f"Workflow error: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
                    st.session_state.state.setdefault("errors", []).append(f"Workflow error: {str(e)}")
    
    # Display PICO
    if st.session_state.state.get('pico'):
        st.header("PICO Framework")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Population:**", st.session_state.state['pico'].get('population', ''))
            st.write("**Intervention:**", st.session_state.state['pico'].get('intervention', ''))
        with col2:
            st.write("**Comparison:**", st.session_state.state['pico'].get('comparison', ''))
            st.write("**Outcome:**", st.session_state.state['pico'].get('outcome', ''))
    
    # Display Articles
    if st.session_state.state.get('articles'):
        st.header("Found Articles")
        for i, article in enumerate(st.session_state.state['articles']):
            with st.expander(f"Article {i+1}: {article['title']}", expanded=False):
                st.write("**Abstract:**", article['snippet'])
                st.write("**URL:**", article['url'])
    
    # Display included articles
    if st.session_state.state.get('included_articles'):
        st.header("Included Articles")
        for i, article in enumerate(st.session_state.state['included_articles']):
            with st.expander(f"Included {i+1}: {article['title']}", expanded=False):
                st.write("**Abstract:**", article['snippet'])
                st.write("**URL:**", article['url'])
    
    # Human Review Extraction
    if st.session_state.state['stage'] == 'review_extracted':
        st.header("Review Extracted Data")
        st.write("Please review the data extracted from studies:")
        
        for i, study in enumerate(st.session_state.state["extracted_data"]):
            with st.expander(f"Study {i+1}: {study['title']}", expanded=True):
                st.write("**Quality:**", f"{study['quality']['risk']} risk - {study['quality']['reason']}")
                
                # Create a clean data display
                data = study['data']
                data_cols = st.columns(2)
                with data_cols[0]:
                    st.write("**Patients:**", data.get('patients', 'N/A'))
                    st.write("**Intervention outcome:**", data.get('intervention_outcome', 'N/A'))
                    st.write("**Comparison outcome:**", data.get('comparison_outcome', 'N/A'))
                with data_cols[1]:
                    st.write("**Effect size:**", data.get('effect_size', 'N/A'))
                    st.write("**Variance:**", data.get('variance', 'N/A'))
                    st.write("**Missing data:**", data.get('missing_data', 'N/A'))
                
                # Allow editing if needed
                if st.checkbox(f"Edit study {i+1} data", key=f"edit_{i}"):
                    with st.form(key=f"edit_form_{i}"):
                        new_patients = st.number_input("Patients", 
                                                      value=int(data.get('patients', 0)),
                                                      min_value=0)
                        new_intervention = st.number_input("Intervention outcome",
                                                          value=float(data.get('intervention_outcome', 0)))
                        new_comparison = st.number_input("Comparison outcome",
                                                        value=float(data.get('comparison_outcome', 0)))
                        new_effect = st.number_input("Effect size",
                                                    value=float(data.get('effect_size', 0)))
                        new_variance = st.number_input("Variance",
                                                      value=float(data.get('variance', 0)))
                        new_missing = st.text_input("Missing data",
                                                   value=data.get('missing_data', ''))
                        
                        if st.form_submit_button("Update Study Data"):
                            st.session_state.state['extracted_data'][i]['data']['patients'] = int(new_patients)
                            st.session_state.state['extracted_data'][i]['data']['intervention_outcome'] = float(new_intervention)
                            st.session_state.state['extracted_data'][i]['data']['comparison_outcome'] = float(new_comparison)
                            st.session_state.state['extracted_data'][i]['data']['effect_size'] = float(new_effect)
                            st.session_state.state['extracted_data'][i]['data']['variance'] = float(new_variance)
                            st.session_state.state['extracted_data'][i]['data']['missing_data'] = new_missing
                            st.success(f"Updated study {i+1}")
        
        # Approval buttons
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve Data and Continue"):
                st.session_state.state["human_approval"] = "yes"
                
                with st.spinner("Performing meta-analysis..."):
                    # Manually set up state to continue after human review
                    current_state = st.session_state.state.copy()
                    
                    # Create new workflow graph to avoid branch issues
                    from main import StateGraph, END, ResearchState
                    from main import perform_meta_analysis, generate_visualizations, draft_report, human_review_report
                    
                    # Create a simple linear graph specifically for this part of the workflow
                    meta_workflow = StateGraph(ResearchState)
                    meta_workflow.add_node("perform_meta_analysis", perform_meta_analysis)
                    meta_workflow.add_node("generate_visualizations", generate_visualizations)
                    meta_workflow.add_node("draft_report", draft_report)
                    meta_workflow.add_node("human_review_report", human_review_report)
                    
                    meta_workflow.set_entry_point("perform_meta_analysis")
                    meta_workflow.add_edge("perform_meta_analysis", "generate_visualizations")
                    meta_workflow.add_edge("generate_visualizations", "draft_report")
                    meta_workflow.add_edge("draft_report", "human_review_report")
                    meta_workflow.add_edge("human_review_report", END)
                    
                    # Compile and run this simpler graph
                    meta_graph = meta_workflow.compile()
                    
                    try:
                        # Run the new focused workflow
                        result_state = meta_graph.invoke(current_state)
                        st.session_state.state = result_state
                        st.rerun()
                    except Exception as e:
                        st.error(f"Workflow error: {str(e)}")
                        st.session_state.state.setdefault("errors", []).append(f"Graph error: {str(e)}")
        with col2:
            if st.button("Reject Data"):
                st.session_state.state["human_approval"] = "no"
                st.warning("Data rejected. Process halted.")
    
    # Human Review Report
    if st.session_state.state['stage'] == 'review_report':
        st.header("Review Final Report")
        
        # Display report
        with st.expander("Report Content", expanded=True):
            st.markdown(st.session_state.state['report_content'])
        
        # Display visualizations if available
        st.subheader("Visualizations")
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.image("forest_plot.png", caption="Forest Plot")
            with col2:
                st.image("funnel_plot.png", caption="Funnel Plot")
        except Exception as e:
            st.warning(f"Visualizations not available: {e}")
        
        # Approval buttons
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve Report and Generate PDF"):
                st.session_state.state["human_approval"] = "yes"
                
                with st.spinner("Generating final PDF..."):
                    try:
                        # Import the generate_pdf function directly
                        from main import generate_pdf
                        
                        # Create a copy of the current state
                        current_state = st.session_state.state.copy()
                        
                        # Call generate_pdf directly
                        st.info("Generating PDF report...")
                        current_state = generate_pdf(current_state)
                        
                        # Update session state
                        st.session_state.state = current_state
                        
                        if current_state["stage"] == "pdf_generated":
                            st.success("PDF generated successfully!")
                            try:
                                with open("final_report.pdf", "rb") as f:
                                    pdf_bytes = f.read()
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=pdf_bytes,
                                        file_name="neurosurgery_meta_analysis.pdf",
                                        mime="application/pdf"
                                    )
                            except Exception as file_error:
                                st.error(f"PDF file error: {str(file_error)}")
                        else:
                            st.error("PDF generation failed.")
                    except Exception as e:
                        st.error(f"PDF generation error: {str(e)}")
                        import traceback
                        tb = traceback.format_exc()
                        st.session_state.state.setdefault("errors", []).append(f"PDF error: {str(e)}")
                        st.text(tb)
                
        with col2:
            if st.button("Reject Report"):
                st.session_state.state["human_approval"] = "no"
                st.warning("Report rejected. Process halted.")
    
    # Display Meta-Analysis Results
    if st.session_state.state.get('meta_results'):
        st.header("Meta-Analysis Results")
        results = st.session_state.state['meta_results']
        
        # Display main results
        if 'error' not in results:
            st.write(f"**Pooled Effect:** {results.get('pooled_effect', 0):.3f}")
            st.write(f"**95% CI:** [{results.get('ci_low', 0):.3f}, {results.get('ci_upp', 0):.3f}]")
            st.write(f"**I² Statistic:** {results.get('i2', 0)*100:.1f}%")
            
            # Display visualizations in columns
            st.subheader("Visualizations")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Display Forest Plot
                if os.path.exists("forest_plot.png"):
                    st.image("forest_plot.png", caption="Forest Plot", use_column_width=True)
                else:
                    st.warning("Forest plot not available")
            
            with viz_col2:
                # Display Funnel Plot
                if os.path.exists("funnel_plot.png"):
                    st.image("funnel_plot.png", caption="Funnel Plot", use_column_width=True)
                else:
                    st.warning("Funnel plot not available")
            
            # Display PRISMA in full width below
            if os.path.exists("prisma_diagram.png"):
                st.image("prisma_diagram.png", caption="PRISMA Flow Diagram", use_column_width=True)
            else:
                st.warning("PRISMA diagram not available")
        else:
            st.error(f"Meta-analysis failed: {results['error']}")
    
    # Display any errors
    if st.session_state.state.get('errors'):
        st.error("Errors encountered:")
        for error in st.session_state.state['errors']:
            st.write(f"- {error}")

    # Display extracted data with warnings and validation guidance
    if st.session_state.state['stage'] == 'extraction_completed':
        st.header("Data Extraction Results")
        
        # Add explicit validation instructions
        st.warning("**IMPORTANT: Please carefully review the extracted data below before approving.**\n\n" + 
                  "Check for:\n" + 
                  "- Accuracy of numerical values (patient counts, outcomes)\n" +
                  "- Appropriateness of effect size calculations\n" +
                  "- Plausibility of variance values (smaller values mean higher precision)\n" +
                  "- Quality assessment accuracy\n" +
                  "- Missing data that might affect meta-analysis")
        
        # Display each study with improved feedback
        for i, study in enumerate(st.session_state.state["extracted_data"]):
            with st.expander(f"Study {i+1}: {study['title']}", expanded=i==0):
                
                # Display original abstract for reference
                st.subheader("Study Information")
                st.write(f"**URL:** {study.get('url', 'Not available')}")
                
                # Find the original article text if available
                original_text = ""
                for article in st.session_state.state.get("included_articles", []):
                    if article.get("title") == study.get("title"):
                        original_text = article.get("snippet", "")
                        break
                
                if original_text:
                    st.subheader("Original Abstract")
                    st.info(original_text)
                    st.markdown("---")
                
                # Show extracted data with potential issue highlights
                st.subheader("Extracted Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Patients:** ", study["data"].get("patients", 0))
                    st.write("**Intervention Outcome:** ", study["data"].get("intervention_outcome", 0))
                    st.write("**Comparison Outcome:** ", study["data"].get("comparison_outcome", 0))
                    
                    # Flag potentially problematic effect sizes
                    effect_size = study["data"].get("effect_size", 0)
                    if abs(effect_size) > 2:
                        st.write("**Effect Size:** ", f"{effect_size} ⚠️ (Unusually large)")
                    else:
                        st.write("**Effect Size:** ", effect_size)
                    
                    # Flag potentially problematic variances
                    variance = study["data"].get("variance", 0)
                    if variance <= 0:
                        st.write("**Variance:** ", f"{variance} ⚠️ (Invalid - must be positive)")
                    elif variance > 1:
                        st.write("**Variance:** ", f"{variance} ⚠️ (Unusually large)")
                    else:
                        st.write("**Variance:** ", variance)
                
                with col2:
                    quality = study.get("quality", {})
                    risk_color = {"Low": "green", "Moderate": "orange", "High": "red"}.get(quality.get("risk"), "gray")
                    st.write("**Quality/Risk of Bias:** ", f":{risk_color}[{quality.get('risk', 'Unknown')}]")
                    st.write("**Reason:** ", quality.get("reason", "Not provided"))
                    
                    # Display any extraction errors or missing data
                    missing_data = study["data"].get("missing_data", "")
                    if missing_data:
                        st.error(f"**Missing Data:** {missing_data}")
                
                # Allow editing of extracted data
                st.subheader("Edit Data")
                effect_size_input = st.number_input(
                    "Effect Size", 
                    value=float(effect_size),
                    step=0.1,
                    key=f"effect_{i}"
                )
                
                variance_input = st.number_input(
                    "Variance", 
                    value=float(max(0.01, variance)),
                    min_value=0.01,
                    step=0.1,
                    key=f"variance_{i}"
                )
                
                patients_input = st.number_input(
                    "Number of Patients", 
                    value=int(study["data"].get("patients", 0)),
                    min_value=0,
                    step=1,
                    key=f"patients_{i}"
                )
                
                # Update the study data when edited
                if effect_size_input != effect_size or variance_input != variance or patients_input != study["data"].get("patients", 0):
                    st.session_state.state["extracted_data"][i]["data"]["effect_size"] = effect_size_input
                    st.session_state.state["extracted_data"][i]["data"]["variance"] = variance_input
                    st.session_state.state["extracted_data"][i]["data"]["patients"] = patients_input
                    st.success("Data updated")

        # Add approval buttons for extraction_completed stage
        st.markdown("---")
        st.subheader("Data Extraction Review")
        
        # Create columns for the buttons
        approve_col1, approve_col2 = st.columns(2)
        
        with approve_col1:
            if st.button("Approve Extracted Data and Continue", key="approve_extraction"):
                try:
                    st.session_state.state["human_approval"] = "yes"
                    st.session_state.state["stage"] = "review_extracted"
                    
                    with st.spinner("Running meta-analysis and generating visualizations..."):
                        # Import required functions
                        from main import perform_meta_analysis, generate_visualizations, draft_report
                        
                        # Create a copy of the current state
                        current_state = st.session_state.state.copy()
                        
                        # Call functions directly in sequence instead of using LangGraph
                        st.info("Performing meta-analysis...")
                        current_state = perform_meta_analysis(current_state)
                        
                        st.info("Generating visualizations...")
                        current_state = generate_visualizations(current_state)
                        
                        st.info("Drafting report...")
                        current_state = draft_report(current_state)
                        
                        # Update session state
                        current_state["stage"] = "review_report"
                        st.session_state.state = current_state
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing meta-analysis: {str(e)}")
                    import traceback
                    tb = traceback.format_exc()
                    st.session_state.state.setdefault("errors", []).append(f"Meta-analysis error: {str(e)}")
                    st.text(tb)  # Show stack trace for debugging
        
        with approve_col2:
            if st.button("Reject Data and Start Over", key="reject_extraction"):
                st.session_state.state["human_approval"] = "no"
                st.warning("Data extraction rejected. Please modify search criteria or edit extracted data.")

if __name__ == "__main__":
    main() 