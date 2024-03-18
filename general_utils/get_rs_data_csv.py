import argparse
import pandas as pd
import rechtspraak_extractor as rex

def filter_rs(df):
    df = df[df['inhoudsindicatie'].apply(lambda x: len(x.split(' ') if isinstance(x, str) else '') >= 15)]
    df = df[df['full_text'].notna() & (df['full_text'] != '')]
    df = df.drop(['creator', 'date_decision', 'issued', 'zaaknummer', 'type', 'references', 'subject', 'relations', 'procedure', 'hasVersion'], axis=1)
    return df


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak extraction')
    parser.add_argument('--start', type=str, default='', help='Start date in format yyyy-mm-dd')
    parser.add_argument('--end', type=str, default='', help='End date in format yyyy-mm-dd')
    args = parser.parse_args()

    df = rex.get_rechtspraak(max_ecli=100000, sd=args.start, ed=args.end, save_file='n')
    df_metadata = rex.get_rechtspraak_metadata(save_file='n', dataframe=df)
    print(df_metadata)
    # filter data on full text availability and reference summary length > 20
    df_filtered = filter_rs(df_metadata)

    print(df_filtered)
    # save
    df_filtered.to_csv("rechtspraak_metadata.csv", index=False)

# python general_utils\get_rs_data_csv.py --start 2022-01-01 --end 2022-12-31
