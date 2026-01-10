import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# with st.sidebar.header("Knowledge Base Upload"):
#     uploaded_file = st.sidebar.file_uploader("Upload knowledge base CSV", type=["csv"])

def frontend_ui():
    st.markdown("<h1>Smart Ticket Analysis</h1>", unsafe_allow_html=True)
            
    col_main, col_res = st.columns([1.5, 1])
            

    st.subheader("Ticket Entry")
    ticket_text = st.text_area("Description", placeholder="Type the user issue here...")

    analyze_btn = st.button("Analyze Ticket", use_container_width=True)

    if analyze_btn:
        if not ticket_text:
            st.warning("Please enter a ticket description.")
        else:
            with st.spinner("🤖 AI processing in progress..."):
                try:
                    final_content = ticket_text

                    # API Calls
                    rec_response = requests.post(f"{API_URL}/recommend", json={"content": final_content})
                    recommendations = rec_response.json().get("recommendations", [])
                        
                    # Display Results
                    st.divider()
                    st.markdown("**📚 Recommendations:**")
                    if recommendations:
                        for idx, rec in enumerate(recommendations, 1):
                            st.info(f"{idx}. {rec}")
                    else:
                        st.warning("No matches found.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the Backend API.")
