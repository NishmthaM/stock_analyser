import os
import jwt
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends
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
# CONFIG
# =========================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("SECRET_KEY")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

app = FastAPI()

client = Groq(api_key=GROQ_API_KEY)

# =========================================
# CORS
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# DATABASE
# =========================================
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client.stock_chat_db
users_collection = db.users

# =========================================
# SECURITY
# =========================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class User(BaseModel):
    username: str
    password: str

def verify_password(p, h):
    return pwd_context.verify(p, h)

def hash_password(p):
    return pwd_context.hash(p)

def create_token(data: dict):
    data = data.copy()
    data["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# =========================================
# STOCK DATA
# =========================================
stocks = {
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS"
}

# =========================================
# ML LOGIC
# =========================================
def load_data(symbol):
    df = yf.download(symbol, period="1y", progress=False)
    if df.empty:
        return None

    df["MA_10"] = df["Close"].rolling(10).mean()
    df.dropna(inplace=True)
    return df

def predict_stock(df):
    try:
        data = df.copy()
        data["lag1"] = data["Close"].shift(1)
        data["Target"] = data["Close"].shift(-1)
        data.dropna(inplace=True)

        X = data[["Close", "MA_10", "lag1"]]
        y = data["Target"]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=50)
        model.fit(X_scaled, y)

        last = scaler.transform(X.iloc[[-1]])
        return model.predict(last)[0], float(df["Close"].iloc[-1])
    except:
        return None, None

def ensemble(symbol):
    df = load_data(symbol)
    if df is None:
        return None

    pred, curr = predict_stock(df)
    if pred is None:
        return None

    change = ((pred - curr) / curr) * 100

    return {
        "current_price": curr,
        "predicted_price": pred,
        "change_percent": change,
        "decision": "BUY" if change > 1 else "SELL" if change < -1 else "HOLD",
        "confidence": min(95, max(55, 100 - abs(change) * 2))
    }

# =========================================
# ROUTES
# =========================================
@app.get("/")
def home():
    return {"status": "Stock Analyzer Running"}

@app.post("/register")
async def register(user: User):
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(400, "User exists")

    await users_collection.insert_one({
        "username": user.username,
        "password": hash_password(user.password)
    })
    return {"msg": "registered"}

@app.post("/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form.username})

    if not user or not verify_password(form.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"sub": form.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict")
async def predict(data: dict, user: str = Depends(get_user)):
    company = data.get("company")

    if company not in stocks:
        return {"error": "Invalid stock"}

    result = ensemble(stocks[company])
    return result

@app.post("/chat")
async def chat(data: dict, user: str = Depends(get_user)):
    msg = data.get("message")

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": msg}]
    )

    return {"response": res.choices[0].message.content}

# =========================================
# IMPORTANT: NO uvicorn.run HERE (Render handles it)
# =========================================