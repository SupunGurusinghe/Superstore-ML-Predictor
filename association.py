import pickle
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# Load the csv file
df = pd.read_csv("storefront.csv")


df_association = df.drop(df.columns[[0, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 17, 19, 20]], axis=1)


#df_association_product
df_association_product = df_association[["ProductName", "ProductID"]].drop_duplicates()
df_association_product = df_association_product.groupby(["ProductName"]).agg({"ProductID":"count"}).reset_index()
df_association_product.sort_values("ProductID", ascending=False).head()


df_association_product.rename(columns={'ProductID': 'ProductID_Count'}, inplace=True)
df_association_product = df_association_product[df_association_product["ProductID_Count"] > 1]


df_association = df_association[~df_association["ProductName"].isin(df_association_product["ProductName"])]


# df_association_product
df_association_product = df_association[["ProductName", "ProductID"]].drop_duplicates()
df_association_product = df_association_product.groupby(["ProductID"]).agg({"ProductName": "count"}).reset_index()
df_association_product.rename(columns={'ProductName': 'ProductName_Count'}, inplace=True)


df_association_product = df_association_product.sort_values("ProductName_Count", ascending=False)
df_association_product = df_association_product[df_association_product["ProductName_Count"] > 1]


# delete product id that represents multiple products
df_association = df_association[~df_association["ProductID"].isin(df_association_product["ProductID"])]
df_California = df[df["State"] == "California"]
df_Florida = df[df["State"] == "Florida"]
df_Texas = df[df["State"] == "Texas"]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['OrderID', 'ProductID'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['OrderID', 'ProductName'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


gr_inv_pro_df = create_invoice_product_df(df_association, id=True)


def check_id(dataframe, productid):
    product_name = dataframe[dataframe["ProductID"] == productid]["ProductName"].unique()[0]
    return productid, product_name


# Determination of Association Rules
frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.0002, use_colnames = True,max_len = 2)


rules = association_rules(frequent_itemsets, min_threshold=0.001)
rules.sort_values("lift", ascending=False).head(10)
sorted_rules = rules.sort_values("lift", ascending=False)

product_id = 'TEC-PH-10003645'
recommendation_list = []

for idx, product in enumerate(sorted_rules["antecedents"]):

    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
            recommendation_list = list(dict.fromkeys(recommendation_list))


