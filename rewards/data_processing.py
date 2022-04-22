import os
import ijson
import pandas as pd

def data_preprocess(filepath):
    '''
    Preprocess human reference data and save result to parquet file format to ./data folder
    Param: 
        filepath: string: Path to the original data file
    '''
    # First import the json data into pandas dataframes
    numbers = [i+3 for i in range (18)] + [22]
    data = []
    columns = [
        "post",
        "split",
        "summary1",
        "summary2",
        "choice"
    ]
    summary1 = ""
    summary2=""
    for num in numbers:
        filename = filepath + str(num) + ".json"
        with open(filename, 'r') as f:
            parser = ijson.parse(f, multiple_values=True)
            chosen_row = []
            summaries = []
            for prefix, event, value in parser:
                if (prefix, event) == ("info.post", "string"):
                    post = value
                    chosen_row.append(post)
                elif (prefix, event) == ("split", "string"):
                    split = value
                    chosen_row.append(split)
                elif (prefix, event) == ("summaries.item.text","string"):
                    if len(summaries) == 2:
                        summaries = []
                        summary1 = value
                        summaries.append(summary1)
                    elif len(summaries) < 2:
                        summary2 = value
                        summaries.append(summary2)
                elif (prefix, event) == ("choice", "number"):
                    choice = value
                    if choice == 1:
                        temp = summary1
                        summary1 = summary2
                        summary2 = temp
                    chosen_row.append(summary1)
                    chosen_row.append(summary2)
                    chosen_row.append(choice)
                    data.append(chosen_row)
                    # Reset
                    chosen_row = []
    df = pd.DataFrame(data, columns=columns)
    os.makedirs('./data')
    df.to_parquet("./data/human_feedback.parquet", engine="pyarrow", index=False)

if __name__== "__main__":
    filepath = "reward_training_data/batch"
    data_preprocess(filepath)
    
    df = pd.read_parquet('data/human_feedback.parquet', engine="pyarrow")
    print(df.dtypes)
    print(df)