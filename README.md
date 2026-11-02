# CPSC480_P3_HousingRegression

Linear regression with gradient descent to predict house prices using size (sq ft) and number of bedrooms. The model predicts about $293,098 for a 1650 sq ft, 3 bedroom house.

## Files

- `Housing_model.py` - the Python script version
- `P3_HousingRegression.ipynb` - the Jupyter Notebook version
- `Housing_data (2).txt` - the training data (size, bedrooms, price)

## Script vs. Notebook

Both have the same code and give the same results. The script is run from the terminal (`python Housing_model.py`) and the plot opens in its own window. The notebook shows the output and plot right under the code, and they're saved in the file so you can see them on GitHub without running it.

One difference: the script uses `__file__` to find the data file, but Jupyter doesn't have `__file__`, so it gave a NameError. I added a try/except that uses the current folder instead when running in Jupyter. Either way, the data file needs to be in the same folder.

Requires numpy, pandas, and matplotlib.
