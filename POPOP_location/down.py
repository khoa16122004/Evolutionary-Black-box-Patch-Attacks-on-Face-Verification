import gdown
output="./"
url = "https://drive.google.com/drive/folders/1OiX3dWd4SxY1mOMhWX_v_aNWdmgWS9CI"
gdown.download(url=url, output=output, fuzzy=True)