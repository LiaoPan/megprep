# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_ace import st_ace  # Import the ACE editor  
from reports.utils import in_docker
from pathlib import Path


# Function to load the Nextflow configuration file
def load_nextflow_conf(file_path):  
    with open(file_path, 'r') as file:  
        content = file.read()  
    return content  

# Function to save the updated content back to the Nextflow configuration file  
def save_nextflow_conf(file_path, content):  
    with open(file_path, 'w') as file:  
        file.write(content)  # 确保 content 是字符串  

def main():
    if in_docker():
        DEFAULT_NXConf_REPORT_FILE = Path("/output") / "nextflow.config"
    else:
        DEFAULT_NXConf_REPORT_FILE = Path(st.session_state.get("dataset_report_path")) / "nextflow.config"

    # Define the path for the nextflow.config file
    config_file_path = Path(st.sidebar.text_input(
        "Nextflow Configure Report Directory:",
        value=DEFAULT_NXConf_REPORT_FILE
    ))

    # Load the configuration  
    config_content = load_nextflow_conf(config_file_path)  

    st.title("Nextflow Configuration Editor")  

    # Create columns for layout  

    # Display the configuration content in the code editor  
    config_content = st_ace(  
            language='hjson',  # Language type for syntax highlighting  
            value=config_content.strip(), 
            auto_update=True,
            keybinding='vscode',
            theme='chaos', 
            height=800,  # Set height of the editor  
        )  


    # Create save and load buttons at the bottom of the page  
    col3, col4 = st.columns(2)  # Create two columns for buttons  

    with col3:  
        if st.button("Save Configuration"):  
            save_nextflow_conf(config_file_path, config_content)  
            st.success("Configuration saved successfully!")  

    with col4:  
        if st.button("Load Current Configuration"):  
            config_content = load_nextflow_conf(config_file_path)  
            st.info("Current configuration loaded.")  

main()
