import pandas as pd

from src.generate_charts import generate


def test_insolvency_chart_generator_handles_partial_data(tmp_path):
    data = tmp_path / "insolvency.csv"
    pd.DataFrame({"period": ["2024-01-01"], "company_insolvencies": [100]}).to_csv(data, index=False)
    generate(tmp_path / "charts", data)
    assert (tmp_path / "charts" / "01_company_insolvency_trend.png").exists()
    assert not (tmp_path / "charts" / "02_individual_insolvency_trend.png").exists()
