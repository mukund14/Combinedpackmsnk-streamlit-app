def func(df):
    # Assuming df is already a pandas DataFrame
    # Process the DataFrame as needed
    print(df.head(3))
    # Perform the rest of the data processing
    # ...

    return {"message": "Processing complete", "data": df.describe().to_dict()}
