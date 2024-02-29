import pandas as pd
from sklearn.cluster import KMeans
import argparse
import os


def main(input_file_path, column_name):
    df = pd.read_csv(input_file_path)

    kmeans = KMeans(n_clusters=2, random_state=0)

    df['Power Cycle'] = kmeans.fit_predict(df[[column_name]])

    cluster_on = df.groupby('Power Cycle')[column_name].mean().idxmax()
    df['Power Cycle'] = df['Power Cycle'].map(
        {cluster_on: 'ON', 1 - cluster_on: 'OFF'})

    target_directory = '/Users/visshal/Elgo-RKA/REFIT_ELGO/CLASSIFICATION'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    output_file_name = f'CLEAN_{os.path.basename(input_file_path).replace("_extracted.csv", "")}_classified.csv'
    output_file_path = os.path.join(target_directory, output_file_name)

    df.to_csv(output_file_path, index=False)
    print(f"File saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classify power cycles of appliances.')
    parser.add_argument('input_file_path', type=str,
                        help='Path to the input CSV file')
    parser.add_argument('column_name', type=str,
                        help='Name of the column to classify (e.g., Fridge-Freezer)')

    args = parser.parse_args()

    main(args.input_file_path, args.column_name)
