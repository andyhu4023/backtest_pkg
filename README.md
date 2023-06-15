# Backtest_pkg

Backtest_pkg is a Python library for backtesting a portfolio strategy or a trading system. 

A portfolio strategy is trying to diverisfy away idiosyncratic risks by constructing a portfolio of optimal weightings. The ideas were originated from the [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory) (MPT) by Harry Markowitz in 1952 and the [Capital Asset Pricing Model](https://en.wikipedia.org/wiki/Capital_asset_pricing_model) (CAPM) from Sharpe in 1964. As the market developed, strategies like statistical arbitrage, smart betas investment were created to capture better risk adjusted returns for portfolio investments. The development of portfolio strategy is still in its early stage and there is huge potentiality in this area. The primary purpose of the package is to provide a simple toolkit to facilitate the development process.

A trading system is trying to buy a security when it's undervalue ande sell it when it's overvalue. The system can be technical (based on only prices and volumes data) or fundamental (rely on financial report figures and comments). Though not a fan for buy-low-sell-high, functionalitis for backtesting a trading system are also provided from completeness. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install backtest_pkg.

```bash
pip install backtest_pkg
```

## Usage

```python
import backtest_pkg

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)