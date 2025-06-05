import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the data
    data = pd.read_excel("data.xlsx")

    # Split the data into training and testing sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the splits to separate files
    train_data.to_excel("train_data.xlsx", index=False)
    test_data.to_excel("test_data.xlsx", index=False)
    
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    