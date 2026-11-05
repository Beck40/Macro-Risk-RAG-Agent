"""Export UK company and individual insolvency trend charts."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def generate(output_dir="assets/charts", data_path="data/insolvency.csv"):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if Path(data_path).exists(): data=pd.read_csv(data_path)
    else:
        data=pd.DataFrame({"period":pd.date_range("2018",periods=24,freq="QE"),"company_insolvencies":[1200+i*30 for i in range(24)],"individual_insolvencies":[9000+i*100 for i in range(24)]})
    data["period"]=pd.to_datetime(data["period"],errors="coerce")
    for column,title,name in [("company_insolvencies","UK company insolvencies","01_company_insolvency_trend.png"),("individual_insolvencies","UK individual insolvencies","02_individual_insolvency_trend.png")]:
        if column not in data: continue
        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(data["period"],data[column],color="#1f77b4"); ax.set(title=title,xlabel="Period",ylabel="Cases"); fig.tight_layout(); fig.savefig(out/name,dpi=160); plt.close(fig)

if __name__=="__main__": generate()

