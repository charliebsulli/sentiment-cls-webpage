import os

# load data from the specified directory
def load_data(dir_path: str) -> tuple[list[str], list[int]]:
    data = []
    labels = []
    for label in ["neg", "pos"]:
        for filename in os.listdir(os.path.join(dir_path, label)):
            file_path = os.path.join(dir_path, label, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
                labels.append(0 if label == "neg" else 1)
    return data, labels