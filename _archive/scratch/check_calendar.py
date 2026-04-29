import pandas as pd
import os

path = r"c:\Users\jermo\Documents\BERLINMAP-DISTANCE\Fahrplan_VBB-2021"
cal = pd.read_csv(os.path.join(path, "calendar.txt"))

# Find weekday services
weekday_services = cal[(cal['monday']==1) & (cal['tuesday']==1) & (cal['wednesday']==1) & (cal['thursday']==1) & (cal['friday']==1)]
print(f"Total services: {len(cal)}")
print(f"Weekday services: {len(weekday_services)}")
print(weekday_services.head())
