import pandas as pd
from pathlib import Path


EXCEL_PATH = Path("data") / "test.xlsx"

print(f"Reading: {EXCEL_PATH.resolve()}")


xls = pd.ExcelFile(EXCEL_PATH)
print("\nAvailable sheets:")
for i, s in enumerate(xls.sheet_names):
    print(f"  [{i}] {s}")


df = pd.read_excel(EXCEL_PATH, sheet_name=0)
print("\nColumns in sheet[0]:")
print(list(df.columns))

print("\nFirst 5 rows (truncated):")
with pd.option_context("display.max_colwidth", 120, "display.width", 120):
    print(df.head(5))
