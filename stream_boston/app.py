import streamlit as st

st.title("👨‍💻 Meet the Developer")

st.markdown("## Hi, I'm Abhinay 👋")

st.markdown("""
### 🚀 Aspiring Data Scientist

I'm an enthusiastic Computer Science student with a strong passion for
**Data Science, Machine Learning, Artificial Intelligence, and Data Analytics**.

I enjoy working with data, discovering meaningful patterns, and developing
practical solutions using Python and modern data science tools.

### 💡 My Interests

- 📊 Data Analysis & Visualization
- 🤖 Machine Learning
- 🧠 Artificial Intelligence
- 🐍 Python Development
- 📈 Business Intelligence
- 🔍 Exploratory Data Analysis

### 🛠️ Tech Stack

`Python` • `Pandas` • `NumPy` • `Matplotlib`

`Seaborn` • `Scikit-Learn` • `Streamlit`

`SQL` • `Power BI`

### 🎯 Career Goal

My goal is to continuously improve my skills in **Data Science and
Machine Learning** and build data-driven solutions that solve
real-world problems.

> *"Turning Data into Decisions and Ideas into Reality."* ✨
""")

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Role", "Aspiring Data Scientist")
c2.metric("Focus", "Data Science")
c3.metric("Specialization", "Machine Learning")
c4.metric("Language", "Python")

st.divider()

st.markdown("""
### 🌱 Always Learning

I believe that becoming a good data scientist is a continuous journey.
I'm constantly learning new technologies, improving my analytical skills,
and working on projects to strengthen my practical knowledge.

### 🤝 Let's Connect

I'm always interested in learning, collaborating, and exploring new
opportunities in **Data Science, Machine Learning, and AI**.
""")

st.success("Thank you for visiting my profile! ❤️")