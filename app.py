import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import re
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Smart Expense Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stTextInput input {
        font-size: 1.1rem;
        padding: 0.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data file path
DATA_FILE = r"C:\Users\HP\Desktop\Langchain\langGraph\transactions_db.json"

# Category keywords for AI parsing
CATEGORY_KEYWORDS = {
    'petrol': 'Fuel', 'diesel': 'Fuel', 'fuel': 'Fuel', 'gas': 'Fuel',
    'tiffin': 'Food', 'lunch': 'Food', 'dinner': 'Food', 'breakfast': 'Food',
    'food': 'Food', 'restaurant': 'Food', 'eat': 'Food', 'meal': 'Food',
    'grocery': 'Groceries', 'vegetables': 'Groceries', 'fruits': 'Groceries',
    'rent': 'Rent', 'house': 'Rent',
    'electricity': 'Utilities', 'water': 'Utilities', 'internet': 'Utilities',
    'phone': 'Utilities', 'mobile': 'Utilities',
    'uber': 'Ride Income', 'ola': 'Ride Income', 'rapido': 'Ride Income',
    'training': 'Training Income', 'freelance': 'Freelance', 'salary': 'Salary',
    'movie': 'Entertainment', 'entertainment': 'Entertainment',
    'medicine': 'Healthcare', 'doctor': 'Healthcare', 'hospital': 'Healthcare',
    'shopping': 'Shopping', 'clothes': 'Shopping', 'electronics': 'Shopping',
    'transport': 'Transport', 'bus': 'Transport', 'metro': 'Transport',
    'education': 'Education', 'course': 'Education', 'books': 'Education',
    'loan': 'Loan', 'borrow': 'Loan', 'lend': 'Loan',
    'interest': 'Interest', 'emi': 'EMI',
    'Other':'Other',
    'Account Balance': 'Account Balance'
}

# Initialize session state
if 'transactions' not in st.session_state:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            st.session_state.transactions = data.get('transactions', [])
            st.session_state.opening_balance = data.get('opening_balance', {})
    else:
        st.session_state.transactions = []
        st.session_state.opening_balance = {}

if 'temp_parsed' not in st.session_state:
    st.session_state.temp_parsed = []

def save_data():
    """Save transactions to JSON file"""
    data = {
        'transactions': st.session_state.transactions,
        'opening_balance': st.session_state.opening_balance
    }
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def parse_transaction(text, trans_type):
    """AI-powered transaction parser"""
    transactions = []
    text = text.lower().strip()
    
    # Pattern: amount + category name (e.g., "220petrol", "50tiffin")
    pattern = r'(\d+\.?\d*)\s*([a-zA-Z\s]+)'
    matches = re.findall(pattern, text)
    
    for amount_str, category_text in matches:
        amount = float(amount_str)
        category_text = category_text.strip()
        
        # Find matching category
        category = 'Other'
        for keyword, cat in CATEGORY_KEYWORDS.items():
            if keyword in category_text:
                category = cat
                break
        
        transactions.append({
            'amount': amount,
            'category': category,
            'description': category_text
        })
    
    return transactions

def get_dataframe():
    """Convert transactions to DataFrame"""
    if not st.session_state.transactions:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.transactions)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_metrics(df, period='all'):
    """Calculate metrics for different time periods"""
    if df.empty:
        return {
            'credit': 0, 'debit': 0, 'borrow': 0,
            'balance': 0, 'transactions': 0, 'opening_balance': 0,
            'closing_balance': 0
        }
    
    # Filter by period
    today = datetime.now()
    if period == 'day':
        df = df[df['date'].dt.date == today.date()]
        month_key = today.strftime('%Y-%m')
    elif period == 'week':
        week_ago = today - timedelta(days=7)
        df = df[df['date'] >= week_ago]
        month_key = today.strftime('%Y-%m')
    elif period == 'month':
        month_ago = today - timedelta(days=30)
        df = df[df['date'] >= month_ago]
        month_key = today.strftime('%Y-%m')
    else:
        month_key = today.strftime('%Y-%m')
    
    credit = df[df['type'] == 'credit']['amount'].sum()
    debit = df[df['type'] == 'debit']['amount'].sum()
    borrow = df[df['type'] == 'borrow']['amount'].sum()
    
    # Get opening balance for current month
    opening = st.session_state.opening_balance.get(month_key, 0)
    closing = opening + credit - debit
    
    return {
        'credit': credit,
        'debit': debit,
        'borrow': borrow,
        'balance': credit - debit,
        'transactions': len(df),
        'opening_balance': opening,
        'closing_balance': closing
    }

def get_financial_advice(df):
    """Generate AI financial advice"""
    if df.empty:
        return "Start tracking your expenses to get personalized financial advice!"
    
    metrics = calculate_metrics(df, 'month')
    advice = []
    
    # Spending analysis
    if metrics['debit'] > metrics['credit'] * 0.8:
        advice.append("⚠️ **High Spending Alert**: You're spending 80%+ of your income. Try to reduce unnecessary expenses.")
    
    # Savings rate
    savings_rate = ((metrics['credit'] - metrics['debit']) / metrics['credit'] * 100) if metrics['credit'] > 0 else 0
    if savings_rate < 20:
        advice.append(f"💡 **Savings Goal**: Current savings rate: {savings_rate:.1f}%. Aim for at least 20% savings.")
    elif savings_rate >= 20:
        advice.append(f"✅ **Great Job**: You're saving {savings_rate:.1f}% of your income! Consider investing surplus funds.")
    
    # Category-wise analysis
    expense_df = df[df['type'] == 'debit']
    if not expense_df.empty:
        category_expenses = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_category = category_expenses.index[0]
        top_amount = category_expenses.iloc[0]
        
        if top_amount > metrics['debit'] * 0.3:
            advice.append(f"📊 **Expense Pattern**: {top_category} is your biggest expense (₹{top_amount:,.0f}). Look for ways to optimize.")
    
    # Investment advice
    if savings_rate > 25:
        advice.append("💰 **Investment Opportunity**: You have good savings! Consider:\n- Mutual Funds (SIP for long-term)\n- Fixed Deposits (for safety)\n- PPF/NPS (for tax benefits)")
    
    # Borrowing alert
    if metrics['borrow'] > metrics['credit']:
        advice.append("🚨 **Debt Alert**: Your borrowings exceed monthly income. Focus on debt repayment first.")
    
    return "\n\n".join(advice) if advice else "✅ Your finances look healthy! Keep tracking regularly."

def calculate_loan_interest(principal, rate, months):
    """Calculate simple interest for loans"""
    interest = (principal * rate * months) / (12 * 100)
    return interest

# Header
st.markdown('<p class="main-header">🤖 Smart AI Expense Manager</p>', unsafe_allow_html=True)
st.markdown("*Intelligent expense tracking with natural language input*")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "➕ Quick Add", "📊 Analytics", "📂 Category Analysis", "💰 Loans", "🤖 AI Advisor", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Quick Stats
    df = get_dataframe()
    today_metrics = calculate_metrics(df, 'day')
    
    st.subheader("Today's Summary")
    st.metric("Credit", f"₹{today_metrics['credit']:,.0f}", delta="Income")
    st.metric("Debit", f"₹{today_metrics['debit']:,.0f}", delta="Expense", delta_color="inverse")
    st.metric("Balance", f"₹{today_metrics['balance']:,.0f}")
    
    # Current month opening balance
    current_month = datetime.now().strftime('%Y-%m')
    opening = st.session_state.opening_balance.get(current_month, 0)
    if opening > 0:
        st.metric("Opening Balance", f"₹{opening:,.0f}", help="Month start balance")
    
    st.divider()
    
    # Export
    if st.button("💾 Export Data", use_container_width=True):
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.subheader("Dashboard Overview")
    
    # Time period tabs
    period_tab = st.radio("View Period", ["Today", "This Week", "This Month", "All Time"], horizontal=True)
    period_map = {'Today': 'day', 'This Week': 'week', 'This Month': 'month', 'All Time': 'all'}
    metrics = calculate_metrics(df, period_map[period_tab])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💚 Total Credit",
            f"₹{metrics['credit']:,.0f}",
            help="All income sources"
        )
    
    with col2:
        st.metric(
            "❤️ Total Debit",
            f"₹{metrics['debit']:,.0f}",
            delta=f"-{(metrics['debit']/metrics['credit']*100) if metrics['credit'] > 0 else 0:.0f}%",
            delta_color="inverse",
            help="All expenses"
        )
    
    with col3:
        st.metric(
            "💙 Net Balance",
            f"₹{metrics['balance']:,.0f}",
            delta=f"{(metrics['balance']/metrics['credit']*100) if metrics['credit'] > 0 else 0:.0f}%",
            help="Credit - Debit"
        )
    
    with col4:
        st.metric(
            "💜 Borrowed",
            f"₹{metrics['borrow']:,.0f}",
            help="Total borrowings"
        )
    
    # Opening and Closing balance
    if metrics['opening_balance'] > 0:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Opening Balance (Month Start):** ₹{metrics['opening_balance']:,.0f}")
        with col2:
            st.success(f"**Net Flow:** ₹{metrics['balance']:,.0f}")
        with col3:
            st.warning(f"**Closing Balance:** ₹{metrics['closing_balance']:,.0f}")
    
    st.divider()
    
    # Charts
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Credit Sources")
            credit_df = df[df['type'] == 'credit'].groupby('category')['amount'].sum().reset_index()
            if not credit_df.empty:
                fig = px.pie(credit_df, values='amount', names='category',
                            color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Debit Categories")
            debit_df = df[df['type'] == 'debit'].groupby('category')['amount'].sum().reset_index()
            if not debit_df.empty:
                fig = px.pie(debit_df, values='amount', names='category',
                            color_discrete_sequence=px.colors.sequential.Reds_r, hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent transactions
        st.subheader("Recent Transactions")
        recent = df.sort_values('date', ascending=True)
        st.dataframe(
            recent[['date', 'type', 'category', 'amount', 'description']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("📝 No transactions yet. Go to 'Quick Add' to start tracking!")

# ==================== QUICK ADD PAGE ====================
elif page == "➕ Quick Add":
    st.subheader("Quick Transaction Entry")
    
    st.info("💡 **How to use**: Type transactions like '220petrol 50tiffin 100lunch' and click Parse!")
    
    # Date and time selection
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("Transaction Date", value=datetime.now())
    with col2:
        selected_time = st.time_input("Transaction Time", value=datetime.now().time())
    
    # Transaction type
    trans_type = st.radio(
        "Select Transaction Type",
        ["Credit", "Debit", "Borrow"],
        horizontal=True,
        help="Credit=Income, Debit=Expense, Borrow=Loan"
    )
    
    # Natural language input
    transaction_input = st.text_area(
        "Enter Transactions",
        placeholder="Example: 220petrol 50tiffin 500uber 100grocery",
        height=100,
        help="Format: amount+category separated by space"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        parse_btn = st.button("🔍 Parse", use_container_width=True, type="primary")
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.temp_parsed = []
            st.rerun()
    
    # Parse transactions
    if parse_btn and transaction_input:
        parsed = parse_transaction(transaction_input, trans_type)
        st.session_state.temp_parsed = parsed
    
    # Display parsed transactions
    if st.session_state.temp_parsed:
        st.success(f"✅ Found {len(st.session_state.temp_parsed)} transactions!")
        
        # Editable table
        for i, trans in enumerate(st.session_state.temp_parsed):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                trans['amount'] = st.number_input(f"Amount #{i+1}", value=trans['amount'], key=f"amt_{i}")
            with col2:
                categories = list(set(CATEGORY_KEYWORDS.values()))
                trans['category'] = st.selectbox(f"Category #{i+1}", categories, 
                                                index=categories.index(trans['category']) if trans['category'] in categories else 0,
                                                key=f"cat_{i}")
            with col3:
                trans['description'] = st.text_input(f"Description #{i+1}", value=trans['description'], key=f"desc_{i}")
            with col4:
                st.write("")
                st.write("")
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.temp_parsed.pop(i)
                    st.rerun()
        
        st.divider()
        
        # Additional fields for borrow type
        if trans_type == "borrow":
            col1, col2, col3 = st.columns(3)
            with col1:
                from_to = st.text_input("Borrowed From / Lent To")
            with col2:
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.5, value=0.0)
            with col3:
                due_date = st.date_input("Due Date", value=datetime.now() + timedelta(days=30))
        
        # Summary
        total = sum(t['amount'] for t in st.session_state.temp_parsed)
        st.info(f"**Total {trans_type.upper()}:** ₹{total:,.0f}")
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("💾 Save All", use_container_width=True, type="primary"):
                for trans in st.session_state.temp_parsed:
                    new_trans = {
                        "id": len(st.session_state.transactions) + 1,
                        "date": selected_date.strftime("%Y-%m-%d"),
                        "time": selected_time.strftime("%H:%M:%S"),
                        "type": trans_type,
                        "category": trans['category'],
                        "amount": trans['amount'],
                        "description": trans['description']
                    }
                    
                    # Add borrow-specific fields
                    if trans_type == "borrow":
                        new_trans["from_to"] = from_to
                        new_trans["interest_rate"] = interest_rate
                        new_trans["due_date"] = due_date.strftime("%Y-%m-%d")
                    
                    st.session_state.transactions.append(new_trans)
                
                save_data()
                st.session_state.temp_parsed = []
                st.success("✅ All transactions saved!")
                st.balloons()
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.temp_parsed = []
                st.rerun()

# ==================== ANALYTICS PAGE ====================
elif page == "📊 Analytics":
    st.subheader("Detailed Analytics")
    
    if not df.empty:
        # Time period selection
        analysis_period = st.selectbox("Analysis Period", ["Daily", "Weekly", "Monthly"], index=2)
        
        # Day-wise, Week-wise, Month-wise analysis
        if analysis_period == "Daily":
            df['period'] = df['date'].dt.date
            group_col = 'period'
        elif analysis_period == "Weekly":
            df['period'] = df['date'].dt.to_period('W').astype(str)
            group_col = 'period'
        else:  # Monthly
            df['period'] = df['date'].dt.to_period('M').astype(str)
            group_col = 'period'
        
        # Credit vs Debit trend
        st.markdown(f"### 📈 {analysis_period} Credit vs Debit Trend")
        trend_data = df[df['type'].isin(['credit', 'debit'])].groupby([group_col, 'type'])['amount'].sum().reset_index()
        
        fig = px.line(trend_data, x='period', y='amount', color='type',
                     markers=True, color_discrete_map={'credit': 'green', 'debit': 'red'})
        fig.update_layout(xaxis_title="Period", yaxis_title="Amount (₹)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💚 Credit Breakdown")
            credit_cat = df[df['type'] == 'credit'].groupby('category')['amount'].sum().sort_values(ascending=False)
            if not credit_cat.empty:
                fig = px.bar(credit_cat, orientation='h', color=credit_cat.values,
                           color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ❤️ Debit Breakdown")
            debit_cat = df[df['type'] == 'debit'].groupby('category')['amount'].sum().sort_values(ascending=False)
            if not debit_cat.empty:
                fig = px.bar(debit_cat, orientation='h', color=debit_cat.values,
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown(f"### 📋 {analysis_period} Summary Table")
        summary = df.groupby([group_col, 'type'])['amount'].sum().reset_index()
        summary_pivot = summary.pivot(index='period', columns='type', values='amount').fillna(0)
        summary_pivot['Balance'] = summary_pivot.get('credit', 0) - summary_pivot.get('debit', 0)
        st.dataframe(summary_pivot.style.format("₹{:,.0f}"), use_container_width=True)
        
    else:
        st.info("No data available for analysis")

# ==================== CATEGORY ANALYSIS PAGE ====================
elif page == "📂 Category Analysis":
    st.subheader("Category-wise Detailed Analysis")
    
    if not df.empty:
        # Time period selection
        period_select = st.selectbox("Select Period", ["Today", "This Week", "This Month", "All Time"])
        period_map = {'Today': 'day', 'This Week': 'week', 'This Month': 'month', 'All Time': 'all'}
        
        # Filter data by period
        today = datetime.now()
        filtered_df = df.copy()
        if period_select == "Today":
            filtered_df = df[df['date'].dt.date == today.date()]
        elif period_select == "This Week":
            week_ago = today - timedelta(days=7)
            filtered_df = df[df['date'] >= week_ago]
        elif period_select == "This Month":
            month_ago = today - timedelta(days=30)
            filtered_df = df[df['date'] >= month_ago]
        
        # Transaction type tabs
        tab1, tab2, tab3 = st.tabs(["💚 Credit Categories", "❤️ Debit Categories", "💜 Borrow Categories"])
        
        with tab1:
            credit_df = filtered_df[filtered_df['type'] == 'credit']
            if not credit_df.empty:
                category_summary = credit_df.groupby('category').agg({
                    'amount': ['sum', 'count', 'mean']
                }).reset_index()
                category_summary.columns = ['Category', 'Total Amount', 'Transactions', 'Average']
                category_summary = category_summary.sort_values('Total Amount', ascending=False)
                
                # Display summary table
                st.markdown("### 📊 Credit Category Summary")
                st.dataframe(
                    category_summary.style.format({
                        'Total Amount': '₹{:,.0f}',
                        'Average': '₹{:,.0f}',
                        'Transactions': '{:.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(category_summary, x='Category', y='Total Amount',
                               title='Total Amount by Category',
                               color='Total Amount',
                               color_continuous_scale='Greens')
                    fig.update_traces(text=category_summary['Total Amount'].apply(lambda x: f'₹{x:,.0f}'),
                                    textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(category_summary, values='Total Amount', names='Category',
                               title='Credit Distribution',
                               color_discrete_sequence=px.colors.sequential.Greens_r,
                               hole=0.4)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed transactions
                st.markdown("### 📋 Detailed Transactions")
                for category in category_summary['Category']:
                    with st.expander(f"💚 {category} - ₹{credit_df[credit_df['category']==category]['amount'].sum():,.0f}"):
                        cat_trans = credit_df[credit_df['category']==category][['date', 'time', 'amount', 'description']].sort_values('date', ascending=False)
                        st.dataframe(cat_trans, use_container_width=True, hide_index=True)
            else:
                st.info("No credit transactions in selected period")
        
        with tab2:
            debit_df = filtered_df[filtered_df['type'] == 'debit']
            if not debit_df.empty:
                category_summary = debit_df.groupby('category').agg({
                    'amount': ['sum', 'count', 'mean']
                }).reset_index()
                category_summary.columns = ['Category', 'Total Amount', 'Transactions', 'Average']
                category_summary = category_summary.sort_values('Total Amount', ascending=False)
                
                # Display summary table
                st.markdown("### 📊 Debit Category Summary")
                st.dataframe(
                    category_summary.style.format({
                        'Total Amount': '₹{:,.0f}',
                        'Average': '₹{:,.0f}',
                        'Transactions': '{:.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(category_summary, x='Category', y='Total Amount',
                               title='Total Amount by Category',
                               color='Total Amount',
                               color_continuous_scale='Reds')
                    fig.update_traces(text=category_summary['Total Amount'].apply(lambda x: f'₹{x:,.0f}'),
                                    textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(category_summary, values='Total Amount', names='Category',
                               title='Expense Distribution',
                               color_discrete_sequence=px.colors.sequential.Reds_r,
                               hole=0.4)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed transactions
                st.markdown("### 📋 Detailed Transactions")
                for category in category_summary['Category']:
                    with st.expander(f"❤️ {category} - ₹{debit_df[debit_df['category']==category]['amount'].sum():,.0f}"):
                        cat_trans = debit_df[debit_df['category']==category][['date', 'time', 'amount', 'description']].sort_values('date', ascending=False)
                        st.dataframe(cat_trans, use_container_width=True, hide_index=True)
            else:
                st.info("No debit transactions in selected period")
        
        with tab3:
            borrow_df = filtered_df[filtered_df['type'] == 'borrow']
            if not borrow_df.empty:
                category_summary = borrow_df.groupby('category').agg({
                    'amount': ['sum', 'count', 'mean']
                }).reset_index()
                category_summary.columns = ['Category', 'Total Amount', 'Transactions', 'Average']
                category_summary = category_summary.sort_values('Total Amount', ascending=False)
                
                # Display summary table
                st.markdown("### 📊 Borrow Category Summary")
                st.dataframe(
                    category_summary.style.format({
                        'Total Amount': '₹{:,.0f}',
                        'Average': '₹{:,.0f}',
                        'Transactions': '{:.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed transactions
                st.markdown("### 📋 Detailed Transactions")
                for category in category_summary['Category']:
                    with st.expander(f"💜 {category} - ₹{borrow_df[borrow_df['category']==category]['amount'].sum():,.0f}"):
                        cat_trans = borrow_df[borrow_df['category']==category][['date', 'time', 'amount', 'from_to', 'interest_rate', 'due_date', 'description']].sort_values('date', ascending=False)
                        st.dataframe(cat_trans, use_container_width=True, hide_index=True)
            else:
                st.info("No borrow transactions in selected period")
    else:
        st.info("No data available for category analysis")

# ==================== LOANS PAGE ====================
elif page == "💰 Loans":
    st.subheader("Loans & Borrowings Management")
    
    if not df.empty:
        loan_df = df[df['type'] == 'borrow'].copy()
        
        if not loan_df.empty:
            st.markdown("### 📊 Active Loans")
            
            for _, loan in loan_df.iterrows():
                with st.expander(f"💰 {loan.get('from_to', 'Unknown')} - ₹{loan['amount']:,.0f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Principal:** ₹{loan['amount']:,.0f}")
                        st.write(f"**Date:** {loan['date'].strftime('%Y-%m-%d')}")
                    
                    with col2:
                        rate = loan.get('interest_rate', 0)
                        st.write(f"**Interest Rate:** {rate}%")
                        due = loan.get('due_date', 'N/A')
                        st.write(f"**Due Date:** {due}")
                    
                    with col3:
                        # Calculate interest
                        if rate > 0 and due != 'N/A':
                            due_date = datetime.strptime(due, '%Y-%m-%d')
                            loan_date = loan['date']
                            months = (due_date.year - loan_date.year) * 12 + (due_date.month - loan_date.month)
                            interest = calculate_loan_interest(loan['amount'], rate, months)
                            total = loan['amount'] + interest
                            st.write(f"**Interest:** ₹{interest:,.2f}")
                            st.write(f"**Total Payable:** ₹{total:,.2f}")
                    
                    st.caption(f"Description: {loan.get('description', 'N/A')}")
            
            # Loan summary
            st.divider()
            total_borrowed = loan_df['amount'].sum()
            st.info(f"**Total Borrowed Amount:** ₹{total_borrowed:,.0f}")
            
            # Interest calculator
            st.markdown("### 🧮 Loan Interest Calculator")
            calc_col1, calc_col2, calc_col3 = st.columns(3)
            
            with calc_col1:
                calc_principal = st.number_input("Principal (₹)", min_value=0, value=10000, step=1000)
            with calc_col2:
                calc_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=2.0, step=0.5)
            with calc_col3:
                calc_months = st.number_input("Months", min_value=1, value=12, step=1)
            
            if st.button("Calculate Interest"):
                interest = calculate_loan_interest(calc_principal, calc_rate, calc_months)
                total = calc_principal + interest
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Interest:** ₹{interest:,.2f}")
                with col2:
                    st.info(f"**Total Amount:** ₹{total:,.2f}")
        else:
            st.info("No loan transactions recorded")
    else:
        st.info("No data available")

# ==================== AI ADVISOR PAGE ====================
elif page == "🤖 AI Advisor":
    st.subheader("AI Financial Advisor")
    
    if not df.empty:
        # Monthly metrics
        monthly_metrics = calculate_metrics(df, 'month')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Credit", f"₹{monthly_metrics['credit']:,.0f}")
        with col2:
            st.metric("Monthly Debit", f"₹{monthly_metrics['debit']:,.0f}")
        with col3:
            savings = monthly_metrics['balance']
            savings_rate = (savings / monthly_metrics['credit'] * 100) if monthly_metrics['credit'] > 0 else 0
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
        
        st.divider()
        
        # Financial advice
        st.markdown("### 💡 Personalized Financial Advice")
        advice = get_financial_advice(df)
        st.markdown(advice)
        
        st.divider()
        
        # Spending patterns
        st.markdown("### 📊 Spending Patterns")
        debit_df = df[df['type'] == 'debit']
        if not debit_df.empty:
            category_spending = debit_df.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            fig = px.treemap(
                names=category_spending.index,
                parents=[""] * len(category_spending),
                values=category_spending.values,
                color=category_spending.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top spending categories
            st.markdown("### 🔝 Top 5 Spending Categories")
            top5 = category_spending.head(5)
            for cat, amt in top5.items():
                percentage = (amt / monthly_metrics['debit'] * 100) if monthly_metrics['debit'] > 0 else 0
                st.write(f"**{cat}:** ₹{amt:,.0f} ({percentage:.1f}%)")
        
        st.divider()
        
        # Investment recommendations
        st.markdown("### 💰 Investment Recommendations")
        
        if savings_rate > 20:
            st.success("✅ You have healthy savings! Consider these options:")
            
            investment_options = {
                "Mutual Funds (SIP)": "Start with ₹2,000-5,000/month for long-term wealth creation",
                "Fixed Deposits": "Safe option with 6-7% returns, good for emergency fund",
                "PPF (Public Provident Fund)": "Tax-free returns ~7%, lock-in 15 years",
                "NPS (National Pension System)": "Retirement planning with tax benefits",
                "Gold": "5-10% portfolio allocation for diversification"
            }
            
            for option, desc in investment_options.items():
                st.write(f"**{option}:** {desc}")
        else:
            st.warning("⚠️ Focus on building savings first before investing")
            st.write("**Tips to increase savings:**")
            st.write("- Track daily expenses diligently")
            st.write("- Cut unnecessary subscriptions")
            st.write("- Cook at home more often")
            st.write("- Use public transport when possible")
        
        # Emergency fund
        st.divider()
        st.markdown("### 🚨 Emergency Fund Status")
        
        emergency_fund_target = monthly_metrics['debit'] * 6  # 6 months expenses
        current_savings = monthly_metrics['balance']
        
        if current_savings >= emergency_fund_target:
            st.success(f"✅ Great! You have sufficient emergency fund (₹{current_savings:,.0f})")
        else:
            needed = emergency_fund_target - current_savings
            st.warning(f"⚠️ Build emergency fund: Need ₹{needed:,.0f} more (Target: 6 months expenses)")
        
    else:
        st.info("Start tracking expenses to get personalized financial advice!")

# ==================== SETTINGS PAGE ====================
elif page == "⚙️ Settings":
    st.subheader("Settings & Configuration")
    
    st.markdown("### 💰 Opening Balance Management")
    st.info("Set the amount you have at the start of each month to track your closing balance accurately")
    
    # Month selection
    col1, col2 = st.columns(2)
    with col1:
        selected_month = st.date_input("Select Month", value=datetime.now(), format="YYYY-MM-DD")
    with col2:
        month_key = selected_month.strftime('%Y-%m')
        current_balance = st.session_state.opening_balance.get(month_key, 0)
        new_balance = st.number_input("Opening Balance (₹)", min_value=0.0, value=float(current_balance), step=1000.0)
    
    if st.button("💾 Save Opening Balance", type="primary"):
        st.session_state.opening_balance[month_key] = new_balance
        save_data()
        st.success(f"✅ Opening balance for {month_key} saved: ₹{new_balance:,.0f}")
    
    st.divider()
    
    # Display all saved opening balances
    st.markdown("### 📊 Saved Opening Balances")
    if st.session_state.opening_balance:
        balance_df = pd.DataFrame([
            {'Month': k, 'Opening Balance': f"₹{v:,.0f}"}
            for k, v in sorted(st.session_state.opening_balance.items(), reverse=True)
        ])
        st.dataframe(balance_df, use_container_width=True, hide_index=True)
    else:
        st.info("No opening balances saved yet")
    
    st.divider()
    
    # Data management
    st.markdown("### 🗄️ Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.checkbox("⚠️ Confirm deletion (this cannot be undone)"):
                st.session_state.transactions = []
                st.session_state.opening_balance = {}
                save_data()
                st.success("✅ All data cleared")
                st.rerun()
    
    with col2:
        total_transactions = len(st.session_state.transactions)
        st.info(f"**Total Transactions:** {total_transactions}")

# Footer
st.divider()
st.caption("🤖 AI-Powered Expense Manager | Made with Streamlit")