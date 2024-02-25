import csv
import requests

csv_file_path = 'test.csv'

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        url = row[0]  # 假设URL在CSV的第一列
        filename = url.split('/')[-1]  # 使用URL的最后一部分作为文件名
        download_file(url, filename)
        print(f"Downloaded {filename} from {url}")