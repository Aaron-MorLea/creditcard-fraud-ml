import shutil
from pathlib import Path
import kagglehub

# 1. Descargar dataset desde Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

# 2. Buscar creditcard.csv
src_dir = Path(path)
src_file = src_dir / "creditcard.csv"

# 3. Copiar a data/raw/ de tu proyecto
dst_file = Path(__file__).resolve().parent / "raw" / "creditcard.csv"
dst_file.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src_file, dst_file)

print("Copiado a:", dst_file)