import pandas as pd
from config import *

df = pd.read_excel(BASE_DIR / "data" / "Telco_customer_churn.xlsx")
df.to_csv(BASE_DIR / "data" / "telco_churn.csv", index=False)

