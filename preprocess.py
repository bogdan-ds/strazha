import pandas as pd


def preprocess_csv(csv_file: str) -> pd.DataFrame:
    # Read the data
    data = pd.read_csv(csv_file, encoding='utf-8-sig')

    # Fill missing values with empty strings
    data['ТЕКСТ'].fillna('', inplace=True)

    # Group by the relevant columns and aggregate the 'Text' column
    merged_df = data.groupby(['ЗАСЕДАНИЕ',
                              'ИЗКАЗВАНЕ',
                              'ГОВОРИТЕЛ',
                              'ПАРЛ_ГРУПА'])['ТЕКСТ'].apply(' '.join)\
        .reset_index()

    # Remove text within brackets from column 'ТЕКСТ'
    merged_df['ТЕКСТ'] = merged_df['ТЕКСТ'].str.replace(r"\(.*\)", "",
                                                        regex=True)
    # In merged_df remove rows where 'ТЕКСТ' is empty or
    # contains less than 12 words
    merged_df = merged_df[merged_df['ТЕКСТ'].str.split().str.len() > 12]

    return merged_df
