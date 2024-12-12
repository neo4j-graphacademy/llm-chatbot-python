import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("Neo4j Chatbot", page_icon=":shopping_cart:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm your Neo4j Chatbot! How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """Submit handler that processes user input and generates response"""
    
    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent with verbose=True
        response = generate_response(message, show_intermediate_steps=True)
        
        print("Response received in bot.py:", response)  # Debug print
        
        # Create an expander for showing the intermediate steps
        if isinstance(response, dict) and 'intermediate_steps' in response:
            print("Found intermediate steps:", response['intermediate_steps'])  # Debug print
            tool_count = 0
            current_tool_steps = []
            
            for step in response['intermediate_steps']:
                print("Processing step:", step)  # Debug print
                # If we see an Action, it's the start of a new tool use
                if 'Action' in step:
                    print("Found Action in step")  # Debug print
                    # If we have previous steps, show them in an expander
                    if current_tool_steps:
                        tool_count += 1
                        print(f"Showing Tool Use #{tool_count}")  # Debug print
                        with st.expander(f"🔧 Tool Use #{tool_count}", expanded=False):
                            for prev_step in current_tool_steps:
                                col1, col2 = st.columns([1, 4])
                                for key, (emoji, label) in {
                                    'Thought': ('🤔', 'Thought'),
                                    'Action': ('⚡', 'Action'),
                                    'Action Input': ('📥', 'Input'),
                                    'Observation': ('👁️', 'Observation')
                                }.items():
                                    if key in prev_step:
                                        col1.markdown(f"**{emoji} {label}:**")
                                        if key == 'Action Input':
                                            col2.markdown(f"```json\n{prev_step[key]}\n```")
                                        elif key == 'Observation':
                                            col2.markdown(f"```\n{prev_step[key]}\n```")
                                        else:
                                            col2.markdown(prev_step[key])
                        current_tool_steps = []
                current_tool_steps.append(step)
            
            # Show any remaining steps
            if current_tool_steps:
                tool_count += 1
                print(f"Showing final Tool Use #{tool_count}")  # Debug print
                with st.expander(f"🔧 Tool Use #{tool_count}", expanded=False):
                    for step in current_tool_steps:
                        col1, col2 = st.columns([1, 4])
                        for key, (emoji, label) in {
                            'Thought': ('🤔', 'Thought'),
                            'Action': ('⚡', 'Action'),
                            'Action Input': ('📥', 'Input'),
                            'Observation': ('👁️', 'Observation')
                        }.items():
                            if key in step:
                                col1.markdown(f"**{emoji} {label}:**")
                                if key == 'Action Input':
                                    col2.markdown(f"```json\n{step[key]}\n```")
                                elif key == 'Observation':
                                    col2.markdown(f"```\n{step[key]}\n```")
                                else:
                                    col2.markdown(step[key])
        else:
            print("No intermediate steps found in response")  # Debug print
        
        # Write the final response outside the expanders
        if isinstance(response, dict) and 'output' in response:
            write_message('assistant', response['output'])
        elif isinstance(response, str):
            write_message('assistant', response)

# Display messages in Session State
for message in st.session_state.messages:
    if message.get('content', '').strip():  # Only display non-empty messages
        write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What would you like to know about movies?"):
    # Display user message in chat message container
    write_message('user', prompt)
    # Generate a response
    handle_submit(prompt)