import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = './home_task_data_set_jan.xltx'
df = pd.read_excel(data_path)

############################################################ Task 1 ############################################################

# Using the percentile function to find the 90th percentile of the classification scores
threshold_score = np.percentile(df['classification_score'], 90)

print()
print('Decline threshold to achieve 90% approval rate: ', threshold_score)

############################################################ Task 2 ############################################################

# Set the style of the visualization
sns.set_style("whitegrid")

# Plotting
plt.figure(figsize=(10, 6))
sns.histplot(df['classification_score'], bins=50, kde=True, color="blue")
plt.axvline(x=threshold_score, color='red', linestyle='--', label=f'Threshold: {threshold_score:.3f}')
plt.title('Distribution of Classification Scores')
plt.xlabel('Classification Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

############################################################ Task 3+4 ############################################################

# Assuming revenue based on fees from approved and chargebacked orders both (CHB/fee*Total_Price=0.5):
def calculate_fee(dataframe):
    sample_df = dataframe.sample(n=10000, random_state=42)

    total_chb_price = sample_df[sample_df['order_status'] == 'chargeback']['price'].sum()

    total_order_price = sample_df['price'].sum()

    fee = 2 * (total_chb_price / total_order_price)

    return fee

fee = calculate_fee(df)
print()
print(f"The calculated fee is: {fee}")

############################################################ Task 5 ############################################################

# Let's look at the data with binary approach:
# Grouping the data by order source to calculate the chargeback rate
order_source_group = df.groupby('order_source')
chargeback_rate_by_source = order_source_group.apply(lambda x: (x.order_status == 'chargeback').mean()).reset_index(name='chargeback_rate')

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='order_source', y='chargeback_rate', data=chargeback_rate_by_source, palette='Set2')
plt.title('Chargeback Rates by Order Source')
plt.xlabel('Order Source')
plt.ylabel('Chargeback Rate')
plt.xticks(rotation=45)
plt.show()

# It seems like there is no significant different between the sources (~0.3%),
# all below 10% chargebacks and the lowest is phone source.

# Let's look at the data in terms of revenue:
# Group by order source and calculate the total price for chargebacks and all orders
totals = df.groupby('order_source').agg(
    total_order_price=('price', 'sum'),
    total_chargeback_price=('price', lambda x: x[df['order_status'] == 'chargeback'].sum())).reset_index()

# Calculate chargeback rate by order source in terms of price
totals['chargeback_rate_price'] = totals['total_chargeback_price'] / totals['total_order_price']

# Plotting the chargeback rates by order source
plt.figure(figsize=(10, 6))
sns.barplot(x='order_source', y='chargeback_rate_price', data=totals, palette='coolwarm')
plt.title('Chargeback Rates by Order Source in Terms of Order Price')
plt.xlabel('Order Source')
plt.ylabel('Chargeback Rate (Price)')
plt.xticks(rotation=45)
plt.show()

# On that approach we can see much significant different,
# and we can say that base on the given dataset, orders that conducted by phone are less risky than the others.

############################################################ Task 6 ############################################################

# Let's examine the risk level of each segment with the chargeback rate from the second approach:
# Calculate the sum of order prices for chargebacks and all orders within each AVS mismatch segment
sums = df.groupby('avs_mismatch').agg(
    total_chargeback_price=('price', lambda x: x[df['order_status'] == 'chargeback'].sum()),
    total_order_price=('price', 'sum')
)

# Calculate the chargeback rate based on order price
sums['chargeback_rate_by_price'] = sums['total_chargeback_price'] / sums['total_order_price']

print()
print(sums[['chargeback_rate_by_price']])

# The difference between the segment is ~1.8%,
# it looks like orders with AVS matching is a little more risky than another.

############################################################ Task 7 ############################################################

## 1. Exploring the relationship between account age and chb rate
# Segment customer accounts into quartiles based on account age
df['account_age_quartile'] = pd.qcut(df['customer_account_age'], 4)

# Generate more descriptive labels based on the quartile ranges
quartile_ranges = df['account_age_quartile'].unique().sort_values()
labels = [f'Q{i+1}: {quartile.left}-{quartile.right} days' for i, quartile in enumerate(quartile_ranges)]
df['account_age_quartile'] = pd.qcut(df['customer_account_age'], 4, labels=labels)

# Calculate CHB rate for each quartile
df['is_chargeback'] = np.where(df['order_status'] == 'chargeback', 1, 0)
chb_rates = df.groupby('account_age_quartile')['is_chargeback'].mean() * 100
chb_rates = chb_rates.reset_index()

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='account_age_quartile', y='is_chargeback', data=chb_rates, palette='coolwarm')
plt.title('CHB Rate by Customer Account Age Quartile')
plt.xlabel('Account Age Quartile (Days)')
plt.ylabel('Chargeback Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# It seems like there is no significant relation between account age and chb rate

## 2. Exploring the relationship between date and chb rate
# Convert 'order_date' to datetime and extract the date component
df['order_date'] = pd.to_datetime(df['order_date']).dt.date

# Create a binary flag for chargebacks
df['is_chargeback'] = df['order_status'] == 'chargeback'

# Group by date and calculate the chargeback rate
daily_chb_rate = df.groupby('order_date')['is_chargeback'].mean() * 100

# Reset index to make plotting easier
daily_chb_rate = daily_chb_rate.reset_index()

# Enhanced Visualization
plt.figure(figsize=(14, 7))
# Plot all daily chargeback rates as a line plot
sns.lineplot(x='order_date', y='is_chargeback', data=daily_chb_rate, label='Daily CHB Rate', color='blue')

# Add a horizontal line at 20% chargeback rate for reference
plt.axhline(y=20, color='red', linestyle='--', label='20% Reference')

# Highlight dates with chargeback rate > 20%
# Filter data to only include days above 20% chargeback rate
above_20 = daily_chb_rate[daily_chb_rate['is_chargeback'] > 20]
# Plot these points specifically
plt.scatter(above_20['order_date'], above_20['is_chargeback'], color='red', label='Above 20%')

plt.title('Daily Chargeback Rate Over Time with 20% Reference Line')
plt.xlabel('Date')
plt.ylabel('Chargeback Rate (%)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# I emphasize all data dots with chb rate above 20%, those dates can be drilled down to check the unusual rate
