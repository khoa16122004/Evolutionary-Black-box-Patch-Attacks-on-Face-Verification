import gdown
output="./"
url = "https://drive.google.com/drive/folders/1CDHDHxG9AYnGs5HHV5IfTNo1V3-YBKmb?usp=sharing"
gdown.download(url=url, output=output, fuzzy=True)
