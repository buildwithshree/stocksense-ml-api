import requests 
s = requests.Session() 
s.headers.update({"User-Agent": "Mozilla/5.0"}) 
t = yf.Ticker("AAPL", session=s) 
df = t.history(period="5d") 
print("rows:", len(df)) 
print(df) 
