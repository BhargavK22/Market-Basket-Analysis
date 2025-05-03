from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import io
import base64
from matplotlib.figure import Figure
import json
import uuid
from datetime import datetime
import random
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pickle
import joblib
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.urandom(24)

os.makedirs('static/models', exist_ok=True)
os.makedirs('static/graphs', exist_ok=True)

df = None
all_products = []
rules_dict = {}
insights = {}
MODEL_PATH = 'static/models/model.pkl'
RULES_PATH = 'static/models/association_rules.pkl'
INSIGHTS_PATH = 'static/models/insights.pkl'
DF_PATH = 'static/models/dataframe.pkl'

@app.before_request
def before_request():
    if 'cart' not in session:
        session['cart'] = []
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    if 'viewed_products' not in session:
        session['viewed_products'] = []

def clean_transaction_data(transactions):
    cleaned_transactions = []
    for transaction in transactions:
        cleaned_transaction = []
        for item in transaction:
            if item is not None and item != '':
                cleaned_transaction.append(str(item).strip())
        if cleaned_transaction:  # Only add non-empty transactions
            cleaned_transactions.append(cleaned_transaction)
    return cleaned_transactions

def load_and_preprocess_data():
    global df, all_products
    
    df = pd.read_csv('ecommerce_data.csv')
    
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    
    df = df.dropna(subset=['Product', 'Product_Category'])
    
    if 'InvoiceNo' not in df.columns:
        df['InvoiceNo'] = df['Customer_Id'].astype(str) + '_' + df['Order_Date'].astype(str)
    
    all_products = df['Product'].unique().tolist()
    
    return df

def prepare_basket_data(df):
    basket = df.groupby(['InvoiceNo'])['Product'].apply(list).reset_index()
    transactions = basket['Product'].tolist()
    
    transactions = clean_transaction_data(transactions)
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_encoded, transactions

def generate_association_rules(df_encoded, min_support=0.01, min_threshold=0.5):
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    rules_dict = {}
    for i, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        antecedents = [str(item) for item in antecedents]
        consequents = [str(item) for item in consequents]
        
        key = ', '.join(antecedents)
        if key not in rules_dict:
            rules_dict[key] = []
        
        rules_dict[key].append({
            'consequent': ', '.join(consequents),
            'support': row['support'],
            'confidence': row['confidence'],
            'lift': row['lift']
        })
    
    return rules, rules_dict

def generate_insights(df):
    insights = {}
    
    top_products = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(10)
    insights['top_products'] = top_products.to_dict()
    
    if 'Order_Date' in df.columns:
        df['Month'] = df['Order_Date'].dt.strftime('%Y-%m')
        monthly_sales = df.groupby('Month')['Sales'].sum().to_dict()
        insights['monthly_sales'] = monthly_sales
    
    if 'Customer_Id' in df.columns:
        customer_spending = df.groupby('Customer_Id')['Sales'].sum().sort_values(ascending=False).head(10)
        insights['top_customers'] = customer_spending.to_dict()
    
    if 'Product_Category' in df.columns:
        category_sales = df.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False).to_dict()
        insights['category_sales'] = category_sales
    
    if 'Gender' in df.columns:
        gender_sales = df.groupby('Gender')['Sales'].sum().sort_values(ascending=False).to_dict()
        insights['gender_sales'] = gender_sales
    
    if 'Device_Type' in df.columns:
        device_sales = df.groupby('Device_Type')['Sales'].sum().sort_values(ascending=False).to_dict()
        insights['device_sales'] = device_sales
    
    if 'Payment_method' in df.columns:
        payment_sales = df.groupby('Payment_method')['Sales'].sum().sort_values(ascending=False).to_dict()
        insights['payment_sales'] = payment_sales
    
    return insights

def get_product_prices(df):
    if 'Sales' in df.columns and 'Product' in df.columns:
        product_prices = df.groupby('Product')['Sales'].mean().to_dict()
        return product_prices
    return {}

def get_popular_products(df, n=4):
    popular_products = []
    
    if df is not None and 'Product' in df.columns:
        top_products = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(n*2)
        
        for i, (product, quantity) in enumerate(top_products.items()):
            if i >= n:
                break
                
            price = 0
            if 'Sales' in df.columns:
                price = df[df['Product'] == product]['Sales'].mean()
                
            description = "Premium quality product with excellent customer ratings."
            
            if i % 4 == 0:
                description = "Best-selling item with exceptional value for money."
            elif i % 4 == 1:
                description = "Customer favorite with outstanding quality and durability."
            elif i % 4 == 2:
                description = "Top-rated product known for its premium features."
            
            rating = round(4.0 + random.random(), 1)
            
            popular_products.append({
                'name': product,
                'price': price,
                'description': description,
                'rating': rating
            })
    
    return popular_products

def generate_dummy_recommendations(n=3):
    dummy_products = [
        {"name": "Premium Wireless Headphones", "confidence": 0.925, "based_on": "your shopping patterns"},
        {"name": "Ultra HD Smart TV", "confidence": 0.897, "based_on": "trending now"},
        {"name": "Professional DSLR Camera", "confidence": 0.873, "based_on": "customers also bought"},
        {"name": "Ergonomic Office Chair", "confidence": 0.856, "based_on": "popular in your area"},
        {"name": "Smart Home Security System", "confidence": 0.842, "based_on": "seasonal favorites"},
        {"name": "Fitness Smartwatch", "confidence": 0.831, "based_on": "your browsing history"}
    ]
    
    random.shuffle(dummy_products)
    
    recommendations = []
    for i in range(min(n, len(dummy_products))):
        recommendations.append({
            'based_on': dummy_products[i]["based_on"],
            'recommendation': dummy_products[i]["name"],
            'confidence': dummy_products[i]["confidence"]
        })
    
    return recommendations

@app.route('/')
def index():
    global df, all_products, rules_dict, insights
    
    if os.path.exists(DF_PATH) and os.path.exists(RULES_PATH) and os.path.exists(INSIGHTS_PATH):
        df = joblib.load(DF_PATH)
        rules_dict = joblib.load(RULES_PATH)
        insights = joblib.load(INSIGHTS_PATH)
    else:
        df = load_and_preprocess_data()
        
        df_encoded, transactions = prepare_basket_data(df)
        
        rules, rules_dict = generate_association_rules(df_encoded)
        
        insights = generate_insights(df)
        
        joblib.dump(df, DF_PATH)
        joblib.dump(rules_dict, RULES_PATH)
        joblib.dump(insights, INSIGHTS_PATH)
    
    top_products = list(insights.get('top_products', {}).items())[:5]
    
    product_prices = get_product_prices(df)
    
    total_sales = 0
    avg_order_value = 0
    
    if 'Sales' in df.columns:
        total_sales = df['Sales'].sum()
        
        if 'InvoiceNo' in df.columns:
            order_values = df.groupby('InvoiceNo')['Sales'].sum()
            avg_order_value = order_values.mean()
    
    sample_recommendations = []
    if rules_dict:
        sample_keys = list(rules_dict.keys())[:5]
        for key in sample_keys:
            if rules_dict[key]:
                sample_recommendations.append({
                    'product': key,
                    'recommendation': rules_dict[key][0]['consequent'],
                    'confidence': rules_dict[key][0]['confidence']
                })
    
    if not sample_recommendations:
        dummy_recs = generate_dummy_recommendations(3)
        for rec in dummy_recs:
            sample_recommendations.append({
                'product': rec['based_on'],
                'recommendation': rec['recommendation'],
                'confidence': rec['confidence']
            })
    
    return render_template('index.html', 
                          top_products=top_products,
                          recommendations=sample_recommendations,
                          product_prices=product_prices,
                          total_sales=total_sales,
                          avg_order_value=avg_order_value,
                          all_products=all_products)

@app.route('/insights')
def show_insights():
    global insights
    
    visualizations = {}
    
    if 'top_products' in insights:
        top_products = dict(list(insights['top_products'].items())[:10])
        fig = px.bar(
            x=list(top_products.keys()),
            y=list(top_products.values()),
            title='Top 10 Products by Sales Volume',
            labels={'x': 'Product', 'y': 'Quantity Sold'},
            color=list(top_products.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        visualizations['top_products'] = fig.to_html(full_html=False)
    
    if 'monthly_sales' in insights:
        monthly_sales = insights['monthly_sales']
        fig = px.line(
            x=list(monthly_sales.keys()),
            y=list(monthly_sales.values()),
            title='Monthly Sales Trend',
            labels={'x': 'Month', 'y': 'Sales Amount'},
            markers=True
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_traces(line=dict(color='#4f46e5', width=3))
        visualizations['monthly_sales'] = fig.to_html(full_html=False)
    
    if 'category_sales' in insights:
        category_sales = dict(list(insights['category_sales'].items())[:10])
        fig = px.pie(
            values=list(category_sales.values()),
            names=list(category_sales.keys()),
            title='Sales by Product Category',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        visualizations['category_sales'] = fig.to_html(full_html=False)
    
    if 'gender_sales' in insights:
        gender_sales = insights['gender_sales']
        fig = px.pie(
            values=list(gender_sales.values()),
            names=list(gender_sales.keys()),
            title='Sales by Gender',
            color_discrete_sequence=['#4f46e5', '#f97316', '#10b981']
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        visualizations['gender_sales'] = fig.to_html(full_html=False)
    
    if 'device_sales' in insights:
        device_sales = dict(list(insights['device_sales'].items())[:10])
        fig = px.bar(
            x=list(device_sales.keys()),
            y=list(device_sales.values()),
            title='Sales by Device Type',
            labels={'x': 'Device Type', 'y': 'Sales Amount'},
            color=list(device_sales.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        visualizations['device_sales'] = fig.to_html(full_html=False)
    
    if 'payment_sales' in insights:
        payment_sales = dict(list(insights['payment_sales'].items())[:10])
        fig = px.bar(
            x=list(payment_sales.keys()),
            y=list(payment_sales.values()),
            title='Sales by Payment Method',
            labels={'x': 'Payment Method', 'y': 'Sales Amount'},
            color=list(payment_sales.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        visualizations['payment_sales'] = fig.to_html(full_html=False)
    
    return render_template('insights.html', visualizations=visualizations, insights=insights)

@app.route('/recommendations')
def recommendations():
    global rules_dict, df
    
    viewed_products = session.get('viewed_products', [])
    
    recommendations = []
    for product in viewed_products:
        if product in rules_dict:
            for rec in rules_dict[product][:2]:  # Get top 2 recommendations for each viewed product
                recommendations.append({
                    'based_on': product,
                    'recommendation': rec['consequent'],
                    'confidence': rec['confidence']
                })
    
    if not recommendations and rules_dict:
        popular_keys = list(rules_dict.keys())[:5]
        for key in popular_keys:
            if rules_dict[key]:
                recommendations.append({
                    'based_on': 'Popular item: ' + key,
                    'recommendation': rules_dict[key][0]['consequent'],
                    'confidence': rules_dict[key][0]['confidence']
                })
    
    if not recommendations:
        recommendations = generate_dummy_recommendations(5)
    
    popular_products = get_popular_products(df)
    
    return render_template('recommendations.html', 
                          recommendations=recommendations,
                          popular_products=popular_products)

@app.route('/product/<product_name>')
def product_detail(product_name):
    global df, rules_dict
    
    viewed_products = session.get('viewed_products', [])
    if product_name not in viewed_products:
        viewed_products.append(product_name)
        session['viewed_products'] = viewed_products
    
    product_data = df[df['Product'] == product_name]
    
    recommendations = []
    if product_name in rules_dict:
        for rec in rules_dict[product_name][:5]:  # Get top 5 recommendations
            recommendations.append({
                'product': rec['consequent'],
                'confidence': rec['confidence']
            })
    
    if not recommendations:
        dummy_recs = generate_dummy_recommendations(3)
        for rec in dummy_recs:
            recommendations.append({
                'product': rec['recommendation'],
                'confidence': rec['confidence']
            })
    
    similar_products = get_popular_products(df, n=3)
    
    return render_template('product.html', 
                          product_name=product_name,
                          product_data=product_data.to_dict('records'),
                          recommendations=recommendations,
                          similar_products=similar_products)

@app.route('/add_to_cart/<product_name>')
def add_to_cart(product_name):
    cart = session.get('cart', [])
    cart.append(product_name)
    session['cart'] = cart
    flash(f'{product_name} added to cart!', 'success')
    return redirect(request.referrer or url_for('index'))

@app.route('/cart')
def view_cart():
    global df, rules_dict
    
    cart = session.get('cart', [])
    
    cart_items = []
    cart_total = 0
    
    if df is not None and cart:
        for product in cart:
            product_data = df[df['Product'] == product]
            if not product_data.empty:
                item = product_data.iloc[0].to_dict()
                if 'Sales' in item and 'Quantity' in item:
                    cart_total += item['Sales'] * item['Quantity']
                elif 'Sales' in item:
                    cart_total += item['Sales']
                cart_items.append(item)
    
    recommendations = []
    if cart and rules_dict:
        for product in cart:
            if product in rules_dict:
                for rec in rules_dict[product][:2]:  # Get top 2 recommendations for each cart item
                    if rec['consequent'] not in cart:  # Don't recommend items already in cart
                        recommendations.append({
                            'based_on': product,
                            'recommendation': rec['consequent'],
                            'confidence': rec['confidence']
                        })
        
        if not recommendations:
            popular_keys = list(rules_dict.keys())[:5]
            for key in popular_keys:
                if rules_dict[key] and key not in cart:
                    recommendations.append({
                        'based_on': 'Popular item',
                        'recommendation': rules_dict[key][0]['consequent'],
                        'confidence': rules_dict[key][0]['confidence']
                    })
    
    if not recommendations:
        recommendations = generate_dummy_recommendations(3)
    
    popular_products = get_popular_products(df)
    
    return render_template('cart.html', 
                          cart_items=cart_items, 
                          recommendations=recommendations,
                          popular_products=popular_products,
                          cart_total=cart_total)

@app.route('/checkout')
def checkout():
    cart = session.get('cart', [])
    
    cart_items = []
    total_price = 0
    if df is not None and cart:
        for product in cart:
            product_data = df[df['Product'] == product]
            if not product_data.empty:
                item = product_data.iloc[0].to_dict()
                if 'Sales' in item:
                    item['TotalPrice'] = item['Sales'] * item.get('Quantity', 1)
                    total_price += item['TotalPrice']
                cart_items.append(item)
    
    return render_template('checkout.html', cart_items=cart_items, total_price=total_price)

@app.route('/confirmation')
def confirmation():
    session['cart'] = []
    
    return render_template('confirmation.html', order_id=str(uuid.uuid4()))

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.args.get('query', '') if request.method == 'GET' else request.form.get('query', '')
    results = []
    
    if query and df is not None:
        results = df[df['Product'].str.contains(query, case=False, na=False)]['Product'].unique().tolist()
    
    popular_products = get_popular_products(df)
    
    return render_template('search.html', 
                          query=query, 
                          results=results,
                          popular_products=popular_products)

@app.route('/advanced_insights')
def advanced_insights():
    global df, insights
    
    rfm_data = None
    if df is not None and 'Customer_Id' in df.columns and 'Order_Date' in df.columns and 'Sales' in df.columns:
        snapshot_date = df['Order_Date'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('Customer_Id').agg({
            'Order_Date': lambda x: (snapshot_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'Sales': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        rfm['Customer_Segment'] = 'Regular'
        rfm.loc[rfm['RFM_Score'].str.startswith('5'), 'Customer_Segment'] = 'Champions'
        rfm.loc[rfm['RFM_Score'].str.startswith('4'), 'Customer_Segment'] = 'Loyal Customers'
        rfm.loc[rfm['RFM_Score'].str.startswith('3'), 'Customer_Segment'] = 'Potential Loyalists'
        rfm.loc[rfm['RFM_Score'].str.startswith('2'), 'Customer_Segment'] = 'At Risk'
        rfm.loc[rfm['RFM_Score'].str.startswith('1'), 'Customer_Segment'] = 'Need Attention'
        
        segment_counts = rfm['Customer_Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            title='Customer Segments (RFM Analysis)',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        rfm_data = {
            'visualization': fig.to_html(full_html=False),
            'summary': rfm.groupby('Customer_Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'CustomerID': 'count'
            }).reset_index().to_dict('records')
        }
    
    discount_data = None
    if 'Discount' in df.columns and 'Sales' in df.columns:
        df['Discount_Bin'] = pd.cut(df['Discount'], 
                                   bins=[0, 0.05, 0.1, 0.2, 0.3, 1], 
                                   labels=['0-5%', '5-10%', '10-20%', '20-30%', '30%+'])
        
        discount_analysis = df.groupby('Discount_Bin').agg({
            'Sales': 'mean',
            'Quantity': 'mean',
            'Profit': 'mean'
        }).reset_index()
        
        fig = px.bar(
            discount_analysis,
            x='Discount_Bin',
            y=['Sales', 'Profit'],
            barmode='group',
            title='Impact of Discount on Sales and Profit',
            labels={'value': 'Amount', 'variable': 'Metric'},
            color_discrete_map={'Sales': '#4f46e5', 'Profit': '#10b981'}
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        discount_data = {
            'visualization': fig.to_html(full_html=False),
            'summary': discount_analysis.to_dict('records')
        }
    
    priority_data = None
    if 'Order_Priority' in df.columns:
        priority_analysis = df.groupby('Order_Priority').agg({
            'Sales': 'sum',
            'Quantity': 'sum',
            'Shipping_Cost': 'mean',
            'InvoiceNo': 'count'
        }).reset_index()
        
        priority_analysis.columns = ['Priority', 'Total Sales', 'Total Quantity', 'Avg Shipping Cost', 'Order Count']
        
        fig = px.bar(
            priority_analysis,
            x='Priority',
            y='Total Sales',
            color='Avg Shipping Cost',
            title='Sales by Order Priority',
            labels={'Total Sales': 'Total Sales Amount'},
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        priority_data = {
            'visualization': fig.to_html(full_html=False),
            'summary': priority_analysis.to_dict('records')
        }
    
    popular_products = get_popular_products(df)
    
    return render_template('advanced_insights.html', 
                          rfm_data=rfm_data, 
                          discount_data=discount_data,
                          priority_data=priority_data,
                          popular_products=popular_products)

if __name__ == '__main__':
    app.run(debug=True)