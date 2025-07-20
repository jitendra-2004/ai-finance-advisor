import streamlit as st
import google.generativeai as genai
import textwrap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


API_KEY = 'enter your api key' 
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    
    pass

# --- DATA MANAGEMENT ("NERVE CENTER") ---

def get_market_data():
    """
    Returns a dictionary of current market data relevant to India.
    In a real-world app, this would be fetched from a live API.
    """
    return {
        "nifty_50_value": 23501.10,
        "rbi_repo_rate": 6.50,
        "inflation_rate_cpi": 4.75,
        "date": datetime.now().strftime("%B %d, %Y")
    }

# --- AI & PROMPT ENGINEERING ---

FINANCIAL_EXPERT_PROMPT = textwrap.dedent("""
    You are a SEBI-registered Senior Investment Advisor with 15 years of experience in the Indian financial market. 
    Your expertise includes investment strategies (Mutual Funds, Equity, NPS, PPF, Fixed Deposits), retirement planning, 
    tax optimization under the Indian Income Tax Act (including Section 80C, 80D, etc.), debt management, and market analysis.
    The user has access to several interactive tools in this app, including a Retirement Planner, a Debt Simulator, and a Tax & Asset Allocation center. Refer to these tools when relevant. Always provide:
    
    1.  **Clear, Actionable Advice:** Tailored to the Indian context.
    2.  **Data-Driven Insights:** Use relevant metrics and examples.
    3.  **Risk Assessment:** Clearly state the risk level for all recommendations (Low/Medium/High).
    4.  **Multiple Options:** Provide alternatives (e.g., different mutual fund categories, tax-saving instruments like ELSS vs. PPF).
    5.  **Personalized Suggestions:** Based on the user's age, risk tolerance, income, etc.
    6.  **Regulatory Compliance:** Mention relevant Indian regulations (SEBI, RBI, PFRDA).
    7.  **Step-by-Step Guidance:** Explain how to implement the advice.
    8.  **Current Market Context:** Relate advice to the current Indian economic environment (NIFTY 50, repo rates, inflation).
    
    Format responses with:
    -   **Bold headings** for different sections.
    -   **Bullet points** for recommendations for clarity.
    -   **Risk ratings** (Low/Medium/High) for investment suggestions.
    
    Always conclude with the mandatory disclaimer: "This is for informational purposes only and does not constitute personal financial advice. Consult a SEBI-registered financial advisor before making any investment decisions."
""")

def create_generative_model():
    """Creates and configures the Gemini generative model."""
    try:
        safety_settings = {
            'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL': 'BLOCK_NONE', 'DANGEROUS': 'BLOCK_NONE'
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={"temperature": 0.4, "top_p": 0.95, "top_k": 40, "max_output_tokens": 2048},
            safety_settings=safety_settings
        )
        return model
    except Exception:
        return None

def init_chat_session(model):
    """Initializes the chat session with the system prompt if the model is available."""
    if model:
        try:
            chat = model.start_chat(history=[])
            chat.send_message(FINANCIAL_EXPERT_PROMPT)
            return chat
        except Exception as e:
            st.error(f"Failed to initialize chat session due to an API error: {e}")
            return None
    return None

# --- CORE FINANCIAL LOGIC & CALCULATORS ---

def get_blended_return(risk_tolerance):
    """
    Calculates the expected annual return based on risk tolerance for Indian markets.
    """
    allocations = {
        "Conservative": {"equity": 0.20, "debt": 0.80},
        "Balanced": {"equity": 0.60, "debt": 0.40},
        "Growth": {"equity": 0.85, "debt": 0.15}
    }
    # Historical average returns for Indian asset classes
    returns = {"equity": 0.12, "debt": 0.07} 
    selected_alloc = allocations.get(risk_tolerance, allocations["Balanced"])
    return (selected_alloc["equity"] * returns["equity"]) + (selected_alloc["debt"] * returns["debt"])

def calculate_debt_payoff(debts_df, extra_payment, method):
    """
    Calculates the debt payoff schedule using Snowball or Avalanche method.
    This logic is universal and works with any currency.
    """
    if debts_df.empty or debts_df['Balance'].sum() <= 0:
        return pd.DataFrame(), 0, 0
    
    debts = debts_df.copy()
    
    if method == 'Snowball (Lowest Balance First)':
        debts = debts.sort_values('Balance', ascending=True).reset_index(drop=True)
    else: # Avalanche (Highest Interest First)
        debts = debts.sort_values('Interest Rate', ascending=False).reset_index(drop=True)
        
    schedule = []
    total_interest_paid = 0
    month = 0
    
    balances = debts['Balance'].values.astype(float)
    rates = debts['Interest Rate'].values.astype(float)
    min_payments = debts['Min Payment'].values.astype(float)

    while balances.sum() > 0.01:
        month += 1
        
        interest_this_month = (balances * (rates / 100)) / 12
        balances += interest_this_month
        total_interest_paid += interest_this_month.sum()
        
        payment_pool = min_payments.sum() + extra_payment
        
        # Pay down the debts according to the strategy
        for i in range(len(balances)):
            if balances[i] > 0:
                # Pay at least the minimum, but more if pool allows
                payment = min(payment_pool, balances[i], min_payments[i] if payment_pool > min_payments[i] else payment_pool)
                # This logic needs refinement to correctly apply snowball/avalanche
                # Simplified logic for now:
                # First, pay all minimums
                paid_so_far = 0
                for j in range(len(balances)):
                    if balances[j] > 0:
                        min_pay = min(balances[j], min_payments[j])
                        balances[j] -= min_pay
                        paid_so_far += min_pay
                
                # Apply extra payment to the priority loan
                extra_pool = (min_payments.sum() + extra_payment) - paid_so_far
                
                for j in range(len(balances)):
                     if balances[j]>0 and extra_pool > 0:
                         pay_extra = min(balances[j], extra_pool)
                         balances[j] -= pay_extra
                         extra_pool -= pay_extra


        schedule.append({
            'Month': month,
            'Remaining Balance': balances.sum(),
            'Total Interest Paid': total_interest_paid
        })
        
        if month > 1200: # Safety break for 100 years
            st.error("Calculation timed out. The payoff period seems excessively long.")
            break
            
    payoff_years = month / 12
    return pd.DataFrame(schedule), total_interest_paid, payoff_years

# --- UI MODULES FOR TABS ---

def chat_advisor_ui():
    st.header("💬 AI Advisor Chat")
    
    if not st.session_state.get('chat'):
        st.warning("The chat is currently unavailable. Please ensure you have a valid API key.", icon="⚠️")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask about investments, tax, or retirement...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        market_data = get_market_data()
        user_data = st.session_state.financial_data
        financial_context = f"""
        [User Financial Profile]
        - Age: {user_data['age']}
        - Risk Tolerance: {user_data['risk_tolerance']}
        - Monthly Income: ₹{user_data['income']:,}
        - Monthly Expenses: ₹{user_data['expenses']:,}
        - Current Investments: ₹{user_data['investments']:,}
        - Total Debts: ₹{user_data['debts']:,}

        [Current Indian Market Conditions as of {market_data['date']}]
        - NIFTY 50: {market_data['nifty_50_value']:,}
        - RBI Repo Rate: {market_data['rbi_repo_rate']}%
        - CPI Inflation Rate: {market_data['inflation_rate_cpi']}%
        """
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("The AI advisor is analyzing your profile..."):
                try:
                    full_prompt = f"{financial_context}\n\nUser Question: {prompt}\n\nProvide comprehensive financial guidance based on your role as a SEBI-registered advisor:"
                    response = st.session_state.chat.send_message(full_prompt)
                    full_response = response.text
                except Exception as e:
                    full_response = f"⚠️ **An error occurred.** This might be due to an invalid API key or network issue.\n\nDetails: {str(e)}"
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

def retirement_planner_ui():
    st.header("🏡 Retirement Corpus Planner")
    st.markdown("Project your retirement savings based on EPF contributions and other investments.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        current_age = st.number_input("Your Current Age", 18, 100, st.session_state.financial_data['age'], key="ret_age")
        target_age = st.number_input("Target Retirement Age", current_age + 1, 100, 60, key="ret_target_age")
    with col2:
        current_savings = st.number_input("Current Retirement Savings (₹)", 0, step=10000, value=st.session_state.financial_data['investments'], key="ret_savings")
        monthly_contribution = st.number_input("Your Voluntary Monthly Contribution (₹)", 0, step=1000, value=8000, key="ret_monthly_contrib", help="Your SIPs or other investments apart from EPF.")
    with col3:
        st.markdown("**EPF Contribution**")
        basic_salary = st.number_input("Your Monthly Basic Salary (₹)", 0, step=1000, value=int(st.session_state.financial_data['income'] * 0.5), key="ret_basic_salary", help="EPF is calculated on your Basic + DA. Typically 40-50% of gross salary.")
        employee_epf_pct = st.slider("Employee EPF Rate (%)", 0.0, 25.0, 12.0, 0.5, format="%.1f", key="ret_employee_epf", help="Standard contribution is 12%.")
        employer_epf_pct = st.slider("Employer EPF Rate (%)", 0.0, 25.0, 12.0, 0.5, format="%.1f", key="ret_employer_epf", help="Standard contribution is 12%.")

    if st.button("Project Retirement Corpus", key="ret_project_btn", type="primary"):
        with st.spinner("Calculating your retirement future..."):
            annual_return = get_blended_return(st.session_state.financial_data['risk_tolerance'])
            months_to_grow = (target_age - current_age) * 12
            
            employee_epf_contrib = basic_salary * (employee_epf_pct / 100)
            employer_epf_contrib = basic_salary * (employer_epf_pct / 100)
            total_monthly_contribution = monthly_contribution + employee_epf_contrib + employer_epf_contrib
            
            projection_data = []
            balance = float(current_savings)

            for month in range(1, months_to_grow + 1):
                balance += total_monthly_contribution
                balance *= (1 + (annual_return / 12))
                
                if month % 12 == 0:
                    year = datetime.now().year + (month // 12)
                    projection_data.append({'Year': year, 'Projected Corpus': balance})
            
            st.subheader("Retirement Projection Results")
            st.markdown(f"Based on an estimated annual return of **{annual_return:.2%}** for a **{st.session_state.financial_data['risk_tolerance']}** profile.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Projected Retirement Corpus", f"₹{balance:,.0f}")
            total_contribs = total_monthly_contribution * months_to_grow
            col2.metric("Total Contributions", f"₹{total_contribs:,.0f}")
            growth = balance - current_savings - total_contribs
            col3.metric("Total Investment Growth", f"₹{growth:,.0f}")

            if projection_data:
                df_proj = pd.DataFrame(projection_data)
                fig = px.area(df_proj, x='Year', y='Projected Corpus', title='Your Retirement Corpus Growth Over Time', markers=True)
                fig.update_layout(xaxis_title='Year', yaxis_title='Projected Corpus (₹)')
                st.plotly_chart(fig, use_container_width=True)

def debt_simulator_ui():
    st.header("💳 Debt Payoff Simulator")
    st.markdown("Simulate your debt-free journey using the popular Snowball or Avalanche methods.")

    if 'debts_df' not in st.session_state:
        st.session_state.debts_df = pd.DataFrame({
            'Name': ['Credit Card', 'Personal Loan', 'Car Loan'],
            'Balance': [50000, 300000, 500000],
            'Interest Rate': [36.0, 14.5, 9.8],
            'Min Payment': [2500, 6000, 11000]
        })

    st.subheader("Your Current Debts")
    st.info("You can edit, add, or remove debts in the table below.")
    edited_df = st.data_editor(st.session_state.debts_df, num_rows="dynamic", key="debts_editor")
    st.session_state.debts_df = edited_df

    col1, col2 = st.columns(2)
    with col1:
        extra_payment = st.number_input("Extra Monthly Payment (₹)", 0, step=500, value=5000, key="debt_extra_payment", help="Additional amount you can pay towards debts each month.")
    with col2:
        method = st.selectbox("Payoff Method", ['Avalanche (Highest Interest First)', 'Snowball (Lowest Balance First)'], key="debt_method", help="Avalanche saves more interest; Snowball provides quick psychological wins.")

    if st.button("Simulate Debt Payoff", key="debt_sim_btn", type="primary"):
        if edited_df.empty or edited_df['Balance'].sum() == 0:
            st.warning("Please add at least one debt to run the simulation.")
        else:
            with st.spinner("Running simulations..."):
                schedule_df, total_interest, payoff_years = calculate_debt_payoff(edited_df, extra_payment, method)
                
                st.subheader("Debt Payoff Results")
                col1, col2 = st.columns(2)
                col1.metric("Time to Become Debt-Free", f"{payoff_years:.1f} Years")
                col2.metric("Total Interest Paid", f"₹{total_interest:,.2f}")

                if not schedule_df.empty:
                    fig = px.line(schedule_df, x='Month', y='Remaining Balance', title='Your Debt Reduction Journey')
                    fig.update_layout(xaxis_title='Months from Now', yaxis_title='Total Remaining Debt (₹)')
                    st.plotly_chart(fig, use_container_width=True)

def tax_allocation_ui():
    st.header("📊 Tax & Asset Allocation Center")
    st.markdown("Estimate your tax liability under the new regime and visualize your asset allocation.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Tax Calculator (New Regime)")
        annual_income = st.number_input("Your Annual Taxable Income (₹)", 0, step=10000, value=st.session_state.financial_data['income'] * 12, key="tax_income")

        slabs = [
            (300000, 0.00),    # Up to 3L -> 0%
            (600000, 0.05),    # 3L to 6L -> 5%
            (900000, 0.10),    # 6L to 9L -> 10%
            (1200000, 0.15),   # 9L to 12L -> 15%
            (1500000, 0.20),   # 12L to 15L -> 20%
            (float('inf'), 0.30) # Above 15L -> 30%
        ]
        
        tax = 0
        remaining_income = annual_income
        prev_slab_limit = 0

        if annual_income > 300000:
            for limit, rate in slabs:
                if remaining_income <= 0: break
                taxable_in_slab = min(remaining_income, limit - prev_slab_limit)
                tax += taxable_in_slab * rate
                remaining_income -= taxable_in_slab
                prev_slab_limit = limit

        # Standard Rebate under Section 87A if income <= 7L
        if annual_income <= 700000:
            tax = 0
            st.success("Income is ≤ ₹7,00,000. Full tax rebate under Sec 87A applies. Your tax liability is zero!")

        st.metric("Estimated Annual Income Tax", f"₹{tax:,.0f}")
        if annual_income > 0:
            st.metric("Effective Tax Rate", f"{(tax / annual_income) * 100:.2f}%")

    with col2:
        st.subheader("Your Asset Allocation")
        risk_tolerance = st.selectbox("Your Risk Tolerance", ["Conservative", "Balanced", "Growth"], index=["Conservative", "Balanced", "Growth"].index(st.session_state.financial_data['risk_tolerance']), key="tax_risk_tol")
        
        allocations = {"Conservative": [20, 70, 10], "Balanced": [60, 30, 10], "Growth": [80, 15, 5]}
        labels = ['Equity (Stocks, MFs)', 'Debt (Bonds, PPF, FD)', 'Gold/Others']
        values = allocations[risk_tolerance]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=px.colors.sequential.Aggrnyl)])
        fig.update_layout(title_text=f'Suggested Allocation for a "{risk_tolerance}" Profile')
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Personal Finance Advisor India", layout="wide", initial_sidebar_state="expanded")

    if 'financial_data' not in st.session_state:
        st.session_state.financial_data = {
            'age': 30, 'risk_tolerance': 'Balanced', 'income': 75000,
            'expenses': 45000, 'investments': 500000, 'debts': 850000
        }
    if 'chat' not in st.session_state:
        model = create_generative_model()
        st.session_state.chat = init_chat_session(model)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'debts_df' not in st.session_state:
        st.session_state.debts_df = pd.DataFrame({
            'Name': ['Credit Card', 'Personal Loan', 'Car Loan'], 'Balance': [50000, 300000, 500000],
            'Interest Rate': [36.0, 14.5, 9.8], 'Min Payment': [2500, 6000, 11000]
        })

    st.title("AI Personal Finance Advisor")
    
    with st.sidebar:
        st.header("👤 Your Profile")
        st.info("The AI uses this profile for personalized advice.")
        
        st.session_state.financial_data['age'] = st.number_input("Age", 18, 100, st.session_state.financial_data['age'], key="sidebar_age")
        st.session_state.financial_data['risk_tolerance'] = st.selectbox("Risk Tolerance", ["Conservative", "Balanced", "Growth"], index=["Conservative", "Balanced", "Growth"].index(st.session_state.financial_data['risk_tolerance']), key="sidebar_risk_tol")
        st.session_state.financial_data['income'] = st.number_input("Monthly Gross Income (₹)", 0, step=1000, value=st.session_state.financial_data['income'], key="sidebar_income")
        st.session_state.financial_data['expenses'] = st.number_input("Monthly Expenses (₹)", 0, step=1000, value=st.session_state.financial_data['expenses'], key="sidebar_expenses")
        st.session_state.financial_data['investments'] = st.number_input("Current Investments (₹)", 0, step=10000, value=st.session_state.financial_data['investments'], key="sidebar_investments")
        
        # Update total debt from the editable dataframe
        st.session_state.financial_data['debts'] = st.session_state.debts_df['Balance'].sum()
        st.metric("Total Debt", f"₹{st.session_state.financial_data['debts']:,.0f}")
        
    tab1, tab2, tab3, tab4 = st.tabs(["AI Advisor Chat", "Retirement Planner", "Debt Simulator", "Tax & Asset Allocation"])
    
    with tab1: chat_advisor_ui()
    with tab2: retirement_planner_ui()
    with tab3: debt_simulator_ui()
    with tab4: tax_allocation_ui()

if __name__ == "__main__":
    main()
