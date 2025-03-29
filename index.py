import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from src.components.model import user_input
import json

st.set_page_config(page_title="Portfolio Analytics", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        .main { background-color: #000000; color: #ffffff; }  /* Black background with white text */
        h1, h2, h3, p, label { color: #ffffff; }  /* White text for readability */
        .stSidebar { background-color: #1c1c1c; color: #ffffff; } /* Dark sidebar */
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            background: #222222; /* Dark gray chat background */
            border-radius: 10px;
            border: 1px solid #ffffff;
        }
        .chat-message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background: #444444; /* Medium gray user messages */
            color: #ffffff;
            align-self: flex-end;
        }
        .bot-message {
            background: #ffffff; /* White bot messages */
            color: #000000;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Load JSON Data-
def load_dashboard_data():
    path = "dashboard_data.json"
    with open(path, "r", encoding="utf-8") as file:
        data = file.read().strip()
        if not data:
            raise ValueError(f"Error: The file {path} is empty.")
        return json.loads(data)

# Load Data
data = load_dashboard_data()

# Sidebar Navigation
st.sidebar.title("🔹 Portfolio Dashboard")
page = st.sidebar.radio("Select Section", ["🏠 Home", "🔥 Skills", "📚 Education", "🌍 Location"])


#Home Section

if page == "🏠 Home":
    st.title("📊 Portfolio Analytics Dashboard")
    st.markdown("### Welcome to my interactive portfolio dashboard! 🚀")

    st.markdown("""
        - **Explore my technical skills** with an interactive **heatmap** 
        - **Dive into my educational journey** using data visualizations 
        - **Locate my work & home addresses** on an interactive **map** 
        - **Chat with me about my portfolio!** 💬
    """)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Interface
    st.markdown("### 💬 Chat with Me")
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            user_msg = f'<div class="chat-message user-message">👤 {chat["user"]}</div>'
            bot_msg = f'<div class="chat-message bot-message">🤖 {chat["bot"]}</div>'
            st.markdown(user_msg, unsafe_allow_html=True)
            st.markdown(bot_msg, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Ask me about my portfolio!")
    if user_question:
        response = user_input(user_question)  # Get response from AI model
        bot_reply = response["output_text"]

        # Save chat history
        st.session_state.chat_history.append({"user": user_question, "bot": bot_reply})

        # Display latest chat
        st.markdown(f'<div class="chat-message user-message">👤 {user_question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message bot-message">🤖 {bot_reply}</div>', unsafe_allow_html=True)

# kills Heatmap Section
elif page == "🔥 Skills":
    st.title("🔥 Skills Proficiency Heatmap")

    # Extract Skills Data
    skills_df = pd.DataFrame({
        "Programming Language": data["Skills"][0]["Programming_languages"],
        "Proficiency Level": list(map(int, data["Skills"][0]["level"]))
    })

    # Heatmap Visualization
    fig_skills = px.bar(skills_df, x="Programming Language", y="Proficiency Level", 
                         title="Skills Proficiency", color="Proficiency Level", 
                         color_continuous_scale="viridis")
    
    st.plotly_chart(fig_skills, use_container_width=True)

# Education Section
elif page == "📚 Education":
    st.title("📚 Education Summary")

    edu_df = pd.DataFrame(data["Education"])

    st.dataframe(edu_df, width=800)

    fig_edu = px.bar(edu_df, x="tag", y=["CGPA", "Percentage"], 
                     labels={"tag": "Education Level"},
                     title="Education Scores",
                     color_discrete_sequence=["#3498DB"])
    
    st.plotly_chart(fig_edu, use_container_width=True)

# Location Section
elif page == "🌍 Location":
    st.title("🌍 Work & Address Locations")

    m = folium.Map(location=[10.8, 79.1], zoom_start=10)

    # Add Markers
    for loc in data["Location"]:
        folium.Marker([float(loc["lat"]), float(loc["long"])], 
                      popup=f"{loc['tag']}: {loc['place']}",
                      icon=folium.Icon(color="blue")).add_to(m)

  
    folium_static(m)

st.sidebar.success("✅ Portfolio Dashboard Loaded Successfully!")
