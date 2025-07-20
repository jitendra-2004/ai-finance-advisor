🇮🇳 AI Personal Finance Advisor for India
A comprehensive, AI-powered personal finance management application built with Streamlit and Google's Gemini Pro. This tool is specifically tailored for the Indian financial landscape, providing users with personalized advice, retirement planning, debt management strategies, and tax calculations relevant to India.

✨ Key Features
This application is designed to be an all-in-one financial dashboard:

🤖 AI Advisor Chat: A conversational AI, powered by Gemini, acting as a SEBI-registered Investment Advisor. It provides personalized financial advice based on the user's profile and current Indian market data (NIFTY 50, RBI Repo Rate, CPI Inflation).

🏡 Retirement Corpus Planner: A detailed retirement calculator that projects your future savings based on voluntary contributions and the Indian Employee Provident Fund (EPF) system.

💳 Debt Payoff Simulator: An interactive tool to simulate debt repayment using the Avalanche (highest interest first) or Snowball (lowest balance first) methods. Users can add, edit, and remove their debts to see a clear path to becoming debt-free.

📊 Tax & Asset Allocation Center:

Tax Calculator: Estimates your annual income tax based on the New Indian Tax Regime (FY 2024-25), including the crucial Section 87A rebate.

Asset Allocation: Visualizes a suggested asset allocation (Equity, Debt, Gold) based on your selected risk tolerance.

🚀 Live Demo
[Optional: Insert a GIF or a screenshot of the application in action here. You can use a tool like ScreenToGif or Kap to record your screen.]

🛠️ Tech Stack
Framework: Streamlit

Generative AI: Google Gemini Pro via google-generativeai

Data Manipulation: Pandas & NumPy

Data Visualization: Plotly Express

Language: Python 3.x

⚙️ Setup and Installation
Follow these steps to run the project locally on your machine.

1. Clone the Repository
git clone [https://github.com/jitendra-2004/ai-finance-advisor.git](https://github.com/jitendra-2004/ai-finance-advisor.git)
cd ai-finance-advisor

2. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

(Note: If you don't have a requirements.txt file, you can create one with pip freeze > requirements.txt after installing the libraries below manually.)

pip install streamlit google-generativeai pandas numpy plotly

4. Configure the API Key
The application requires a Google Generative AI API key to function.

Get your API key from Google AI Studio.

In the main.py file, replace the placeholder with your actual API key:

# In main.py
API_KEY = 'YOUR_GOOGLE_AI_API_KEY'

For better security, it is highly recommended to use Streamlit's secrets management. Create a file .streamlit/secrets.toml and add your key:

# .streamlit/secrets.toml
API_KEY = "YOUR_GOOGLE_AI_API_KEY"

Then, in your Python code, you can access it via st.secrets["API_KEY"].

5. Run the Application
Once the dependencies are installed and the API key is configured, run the following command in your terminal:

streamlit run main.py

The application should now be running and accessible in your web browser!
