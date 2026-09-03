# Portfolio Manager — standalone

This folder is a **separate decision-support module**. It does not modify or feed data into the core stock prediction pipeline.

Input: `data/my_portfolio.csv`.

The CSV is designed for the portfolio snapshot supplied by the user: `index,Stock,Quantity,Current_PnL_INR,Return_Percent`.

The manager can report current P&L, multi-horizon outlook (1D/5D/20D/60D/120D/252D), action, target, risk/stop, recovery-to-average percentage, averaging quantity suggestion, and tomorrow profit-booking alert.

No profit is guaranteed. Missing historical purchase price is flagged as estimated rather than silently treated as exact.