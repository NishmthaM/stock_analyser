import os
import asyncio
import jwt
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pydantic import BaseModel, validator
from groq import Groq
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

# =========================================
# 1. CONFIGURATION & ENV
# =========================================
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440 

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)

# =========================================
# 2. CORS & DATABASE
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client.stock_chat_db
users_collection = db.users

# =========================================
# 3. AUTH & SECURITY HELPERS (Must be above routes)
# =========================================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class User(BaseModel):
    username: str
    password: str

    @validator("password")
    def check_password(cls, v):
        if len(v) > 72: raise ValueError("Password too long")
        return v

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# =========================================
# 4. DATA LOGIC & MODELS
# =========================================
stocks = {
    "Trident Ltd": "TRIDENT.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Infosys": "INFY.NS",
    "Reliance Industries": "RELIANCE.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "L&T Technology Services": "LTTS.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS"
}

def load_data(symbol):
    try:
        df = yf.download(symbol, period="2y", progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df['MA_10'] = df['Close'].rolling(10).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)
        return df if len(df) >= 20 else None
    except: return None

def get_news(company):
    try:
        url = "https://newsapi.org/v2/everything"
        headers = {'User-Agent': 'TradeX/1.0'}
        params = {"q": f"{company} stock", "sortBy": "publishedAt", "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
        res = requests.get(url, params=params, headers=headers, timeout=5)
        articles = res.json().get("articles", [])
        return 0, articles
    except: return 0, []

def prophet_model(df):
    try:
        data = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet(daily_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=3)
        forecast = model.predict(future)
        return float(forecast["yhat"].iloc[-1]), float(data["y"].iloc[-1])
    except: return None, None

def ml_model(df):
    try:
        df = df.copy()
        df['lag1'], df['lag2'] = df['Close'].shift(1), df['Close'].shift(2)
        df['returns'] = df['Close'].pct_change()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        X = df[['Close', 'MA_10', 'RSI', 'lag1', 'lag2', 'returns']]
        y = df['Target']
        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_s, y)
        last = scaler.transform(X.iloc[[-1]])
        return model.predict(last)[0]
    except: return None

def ensemble_predict(symbol, company):
    df = load_data(symbol)
    if df is None: return None
    p_pred, current_price = prophet_model(df)
    m_pred = ml_model(df)
    
    preds = [p for p in [p_pred, m_pred] if p is not None]
    if not preds: return None
    
    final_pred = np.mean(preds)
    change = ((final_pred - current_price) / current_price) * 100
    confidence = max(55, min(92, 100 - (df['Close'].pct_change().std() * 200)))

    return {
        "current_price": float(current_price),
        "predicted_price": float(final_pred),
        "change_percent": float(change),
        "decision": "🟢 BUY" if change > 1.5 else "🔴 SELL" if change < -1.5 else "🟡 HOLD",
        "confidence": float(confidence),
        "trend": "Uptrend" if change > 0 else "Downtrend"
    }
@app.post("/chat")
async def chat(data: dict, current_user: str = Depends(get_current_user)):
    user_message = data.get("message")
    if not user_message:
        return {"response": "No message received"}
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": user_message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# =========================================
# 5. ROUTES (Defined last)
# =========================================
@app.post("/register")
async def register(user: User):
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(400, "Username already exists")
    hashed = get_password_hash(user.password)
    await users_collection.insert_one({"username": user.username, "password": hashed})
    return {"message": "User created"}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/news")
async def news(data: dict, current_user: str = Depends(get_current_user)):
    name = data.get("company")
    score, articles = get_news(name)
    return {"articles": articles}

@app.post("/screener")
async def run_screener(data: dict, current_user: str = Depends(get_current_user)):
    # 1. Get filters from the request (defaulting to Streamlit-like values)
    selected_stocks = data.get("stocks", list(stocks.keys()))
    min_conf = data.get("min_conf", 60)
    min_change = data.get("min_change", 2)
    decision_filter = data.get("decision_filter", "ALL")

    results = []

    for name in selected_stocks:
        if name not in stocks:
            continue
            
        symbol = stocks[name]
        
        # Call your prediction engine
        # IMPORTANT: Ensure ensemble_predict returns a dictionary!
        res = ensemble_predict(symbol, name)
        
        if not res:
            continue

        # 2. Extract values for filtering
        # These keys must match what ensemble_predict returns
        confidence = res.get("confidence", 0)
        change = res.get("change_percent", 0)
        decision = res.get("decision", "🟡 HOLD")

        # 3. Apply your Smart Filters
        # Filter by Confidence %
        if confidence < min_conf:
            continue

        # Filter by Minimum Expected Move (Volatility/ROI)
        if abs(change) < min_change:
            continue

        # Filter by Decision Type (BUY/SELL/HOLD)
        if decision_filter != "ALL" and decision != decision_filter:
            continue

        # 4. If it passed all filters, add to results
        results.append({
            "company": name,
            **res # Unpacks all other data (price, trend, etc.)
        })

    # Sort by confidence descending (highest conviction first)
    results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
    
    return {"results": results}

@app.post("/predict")
async def predict(data: dict, current_user: str = Depends(get_current_user)):
    company = data.get("company")
    if company not in stocks:
        return {"error": "Invalid company"}

    # 1. Fetch data and run prediction
    symbol = stocks[company]
    res = ensemble_predict(symbol, company)

    if not res:
        return {"error": "Could not generate prediction for this stock"}

    # 2. Extract values safely from the dictionary returned by ensemble_predict
    # This prevents the "UndefinedVariable" errors
    curr = res.get("current_price", 0)
    pred = res.get("predicted_price", 0)
    change = res.get("change_percent", 0)
    decision = res.get("decision", "🟡 HOLD")
    confidence = res.get("confidence", 0)
    sentiment = res.get("sentiment_score", 0)

    # 3. Fetch OHLC history for the dashboard's candle chart
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1mo", interval="1d")
    
    ohlc_history = []
    for index, row in df.iterrows():
        ohlc_history.append({
            "x": int(index.timestamp() * 1000),
            "y": [
                round(row['Open'], 2), 
                round(row['High'], 2), 
                round(row['Low'], 2), 
                round(row['Close'], 2)
            ]
        })

    # 4. Final return object
    return {
        "current_price": curr,
        "predicted_price": pred,
        "change_percent": change,
        "decision": decision,
        "confidence": confidence,
        "sentiment_score": sentiment,
        "history": ohlc_history
    }

# Ensure this is at the bottom of main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
