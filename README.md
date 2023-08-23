# ES_FiveFactor
Python for LSTM/TF on five time series. Initial results indicate that it's worth investing in making this
approach into a python module, so I've gone ahead and tidied it up.

This takes prices for five asset classes: stocks, US currency, Treasury bonds, gold, and Bitcoin, and inputs
lagged values (one current, and eleven lagged) into a Tensorflow LSTM model. To do that effectively, 
preserving the floating point nature of the pricing data, various bespoke normalizations had to be done, 
including one to preserve a single-mode centralized distribution (without excessive skewness).

The data for this are frankly, insufficient on their own to meaningfully drive an ANN-based model of this 
size, even with in excess of 30k five-minute observations, easily enough to determine parameters, but not to 
avoid overfitting and inapplicable out of sample forecasting ability. Then there is the issue of a massively 
parameterized model picking local optima and never being able to effectively explore the parameter space 
globally, even with the best hyperparameter tuning and randomized descent methods.

To address this, a pool of such models are started, and kept in a fixed-size "gene pool" of them, so to 
speak. The "fitness" criterion is the out of sample fit on the evaluation data set, the data excluded from 
the LSTM model's training set. The training set is always the oldest contiguous time series data, and by far 
the largest. The evaluation data set always comes disjointly right afterwards, up to the concurrent forecast 
support data.

The actual learning therefore takes place in the context of the outer, genetic algorithm, which selects which
of the ANN-based learning results is worthwhile on out of sample data. That inner learning cycle chooses
local optima in a high dimensional parameter space, and cannot be trusted to explore the full space for a 
global optimum without the assistance of innovation and extinction of starting points and input data, by the 
genetic algorithm.

I've obtained out of sample R-squareds in excess of 2% on individual models, which, with a pool 5 of them, 
almost makes this a tradeable system. Obviously, the validity and out of sample performance of models 
changes over time as market conditions change, and new market data are encountered.
