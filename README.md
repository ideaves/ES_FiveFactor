# ES_FiveFactor
Python for LSTM/TF on five time series. Initial results indicate that it's worth investing in making this
approach into a python module, so I've gone ahead and tidied it up. Just drop in *.py, and a csv file with
periodic data (5 minute assumptions may be partly baked in, and require a fix), for about 35000 time periods
formatted like CurrentBars.csv. Decimal or fractional bond prices should work fine.

This takes prices for five asset classes: stocks (E-mini futures), US currency, Treasury bonds, gold, and 
Bitcoin, and inputs lagged values (one current, and eleven lagged) into a Tensorflow LSTM model. To do that 
effectively, preserving the floating point nature of the pricing data, a bespoke normalization had to be 
done, using asymmetric Winsorization to create an output range symmetric around zero, for the supervisory 
series.

The data for this are frankly, insufficient on their own to meaningfully drive an ANN-based model of this 
size, even with in excess of 30k five-minute observations, easily enough to determine parameters, but not to 
avoid overfitting and inapplicable out of sample forecasting ability. Then there is the issue of a massively 
parameterized model picking local optima and never being able to effectively explore the parameter space 
globally, even with the best hyperparameter tuning and randomized descent methods.

To address this, a pool of such models are started, and kept in a fixed-size "gene pool" of them, so to 
speak. The "fitness" criterion is the out of sample fit on the evaluation data set, the data excluded from 
the LSTM model's training set. The fitness criterion allows for risk aversity with respect to variability of 
the out of sample performances, and with respect to the opportunity cost of keeping a model, when there is a 
fixed gene pool size and there may be opportunities constrained to be left unexplored. The training set is 
always the oldest contiguous time series data, and by far the largest. The evaluation data set always comes 
disjointly right afterwards, up to the concurrent forecast support data.

The actual learning therefore takes place in the context of the outer, genetic algorithm, which selects which
of the ANN-based learning results is worthwhile on out of sample data. That inner learning cycle chooses
local optima in a high dimensional parameter space, and cannot be trusted to explore the full space for a 
global optimum without the assistance of innovation and extinction of starting points and input data, by the 
genetic algorithm.

I've obtained out of sample R-squareds in excess of 2% on individual models, which, with a pool 5 of them, 
almost makes this a tradeable system. Obviously, the validity and out of sample performance of models 
changes over time as market conditions change, and new market data are encountered.

Directions for future research include a bigger and more highly powered pool of models of course. But more
interestingly, an exploration of including a separate model for a look into the future output basis series, 
optimizing those parameters alongside those of the predictive model. In effect, you'd be simultaneously 
searching for whatever range or shape of prediction can be found that predicts most consistently on out of 
sample data.

Training note: If you've downloaded this in order to assess the methodology, please be aware that it was 
intended to train on live data. Every training, evaluation, and mutation selection cycle should encounter
a set of market data from at least one time period forward from the prior cycle. Nothing will be harmed if
that is not the case, but gradually the genetic algorithm wrapper will overfit for any evaluation data set
that remains unchanged (ordering must be preserved, as it is a time series model - data randomization is  
not an option). Currently, this cycles though five models about every 15-25 minutes using the CUDA in an 
older desktop from about 2018. The market data is at five minute intervals, so this works nicely.
