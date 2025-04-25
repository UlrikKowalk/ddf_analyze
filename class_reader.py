import pandas as pd


def read_audacity_labels(file_path):
    """
    Reads an Audacity label track file and returns it as a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the Audacity label file (.txt)

    Returns:
    - DataFrame with columns ['start', 'end', 'label']
    """
    try:
        # Read the tab-separated file
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['start', 'end', 'label'],
            engine='python',
            quoting=3,  # prevent issues with quotes in labels
            skip_blank_lines=True
        )

        return df

    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    path_to_file = "mumie_seite_A.txt"  # Change to your file path
    labels_df = read_audacity_labels(path_to_file)

    if labels_df is not None:
        print(labels_df)