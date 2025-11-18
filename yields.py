import matplotlib.pyplot as plt

mats = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
us_yields = [3.89, 3.79, 3.7, 3.6, 3.6, 3.71, 3.89, 4.11, 4.65, 4.67]
eur_yields = [1.89, 1.90, 1.92, 1.99, 2.08, 2.29, 2.50, 2.78, 3.25, 3.28]

plt.figure(figsize=(10, 6))
plt.plot(mats, us_yields, marker='o', label='US Yields', color='blue')
plt.plot(mats, eur_yields, marker='o', label='EUR Yields', color='orange')
plt.title('US vs EUR Yield Curves')
plt.xlabel('Maturities (Years)')
plt.ylabel('Yields (%)')
plt.xticks(mats, ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'])
plt.show()