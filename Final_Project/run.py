import streamlit as st
from Frontend.frontend import frontend_ui

#session starting
if 'logged_in' not in st.session_state:
    st.session_state.session = "user_1"

# Main Routing
if st.session_state.session:
    frontend_ui()
