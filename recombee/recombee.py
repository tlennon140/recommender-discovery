# https://medium.com/recombee-blog/personalized-recommendations-in-10-minutes-bcbea144974b

'''

How to use Recombee:
    1. Upload product catalogue to corresponding database
    2. Collect data about the specific user from the survey
    3. Use the code below to create recommendations

'''

from recombee_api_client.api_client import RecombeeClient
from recombee_api_client.api_requests import *
import csv

# Upload data from which to build recommendations
purchases = []
with open('purchases.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        user_id = row[0]
        item_id = row[1]
        time = row[2]
        r = AddPurchase(user_id, item_id,
                        timestamp=time, cascade_create=True)
        purchases.append(r)


client = RecombeeClient(--db--, --secret--)

br = Batch(purchases)
client.send(br)

r = AddDetailView(user_id, item_id,
                  timestamp=time, cascade_create=True)

# Create recommendations for user-27 - user based are the default recommendations based on user data in DB; 
# item based recommendations are once the user clicks on a product, to recommend similar products
recommended = client.send(UserBasedRecommendation('user-27', 5))
print(recommended)

recommended = client.send(
   ItemBasedRecommendation('event-32', 5, target_user_id='user-27'))