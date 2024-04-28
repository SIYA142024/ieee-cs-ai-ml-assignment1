import pandas as pd
url = "https://drive.google.com/uc?id=1CEqI-OEexf9p02M5vCC1RDLXibHYE9Xz"
data = pd.read_csv(url)
print(data.head())
