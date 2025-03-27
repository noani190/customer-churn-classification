import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PATH

data = pd.read_csv(DATA_PATH)

print(f'plan type: {data.plan_type.value_counts()}')
print(f'classes: {data.churn.value_counts()}')

data.plan_type.value_counts().sort_values().plot(kind = 'bar')
plt.savefig('visualizations/overall_plan_type.png', bbox_inches='tight')
plt.close()

data[['transaction_amount', 'date']].boxplot(by='date')
plt.xticks(rotation=90)
plt.savefig('visualizations/transactions_by_date.png', bbox_inches='tight')
plt.close()

data[['transaction_amount', 'customer_id']].boxplot(by='customer_id')
plt.xticks(rotation=90)
plt.savefig('visualizations/transactions_by_customer.png', bbox_inches='tight')
plt.close()

for customer_id, group in data.groupby('customer_id'):
    if int(customer_id.split('_')[1]) < 10:
        group.plot(x='date', y='transaction_amount', title=f'Customer ID: {customer_id}')
        plt.savefig(f'visualizations/transactions_by_customer_{customer_id}.png', bbox_inches='tight')
        plt.close()
