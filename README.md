# iTransformer Engine

Inverted Transformer that treats each ETF and macro variable as a token (channel‑independent). The model learns dependencies across variables via self‑attention. Conformal prediction provides 90% prediction intervals; the upper bound is used as an optimistic score. Multi‑window selection (63–2016 days) picks the best window per ETF.

- **iTransformer:** Embedding, Transformer encoder, readout
- **Variable selection:** All variables are used; macro variables condition the forecast
- **Conformal prediction:** Upper bound of 90% prediction interval
- **Output:** top 3 ETFs per universe by optimistic score

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
