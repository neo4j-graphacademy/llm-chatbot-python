import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# tag::write_message[]
def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)
# end::write_message[]

# tag::get_session_id[]
def get_session_id():
    """Get the current session ID"""
    ctx = get_script_run_ctx()
    if ctx is None:
        return None
    return ctx.session_id
# end::get_session_id[]