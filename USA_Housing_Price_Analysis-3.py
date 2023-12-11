# Libraries
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.sql.functions import mean, split, col, sum, when
import numpy as np
import boto3
from pyspark.sql.types import IntegerType , FloatType

s3 = boto3.client('s3')

filenames = ['plot1.png', 'plot2.png', 'plot3.png','plot4.png','plot5.png','plot6.png','plot7.png','plot8.png','plot9.png','plot10.png','plot11.png', 'plot12.png']



# Creating a Spark session
spark = SparkSession.builder.appName("DataFrameToRDD").getOrCreate()

# Reading csv file
df = spark.read.csv("s3://usa-housing-price/housing123.csv", header=True, inferSchema=True)

# Data Filtering 
df = df.drop("reviews", "description", "region_url", "image_url", "url", "lat", "long")
df = df.withColumn("region_first", split(col("region"), "/").getItem(0))
df = df.na.drop() 
df = df.withColumn("price", df["price"].cast(IntegerType()))
df = df.withColumn("sqfeet", df["sqfeet"].cast(IntegerType()))
df = df.withColumn("cats_allowed", df["cats_allowed"].cast(IntegerType()))
df = df.withColumn("dogs_allowed", df["dogs_allowed"].cast(IntegerType()))
df = df.withColumn("beds", df["beds"].cast(IntegerType()))
df = df.withColumn("baths", df["baths"].cast(IntegerType()))

#Top 10 cheap regions by average price in USA
# Calculate the average price for each region
avg_price_rdd = (
    df.groupBy("region_first")
    .agg({"price": "avg"})
    .rdd
    .sortBy(lambda x: x[1], ascending=True)  # Sort by average price in ascending order
)

# Take the top 10 regions
top_10_regions = avg_price_rdd.take(10)

# Collect data to the driver for visualization
regions, avg_prices = zip(*top_10_regions)

# Create a bar chart
fig1 = plt.figure(figsize=(15, 8))
plt.bar(regions, avg_prices)
plt.xticks(rotation=45)
plt.title('Top 10 cheap regions by average price in USA')
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.tight_layout()
fig1.savefig('plot1.png', dpi=300)




#Top 10 cheap states by average price in USA
# Calculate the average price for each state
avg_price_states_rdd = (
    df.groupBy("state")
    .agg({"price": "avg"})
    .rdd
    .sortBy(lambda x: x[1], ascending=True)  # Sort by average price in ascending order
)

# Take the top 10 regions
top_10_states = avg_price_states_rdd.take(10)

# Collect data to the driver for visualization
states, avg_prices = zip(*top_10_states)


# Create a bar chart
fig2 = plt.figure(figsize=(15, 8))
plt.bar(states, avg_prices)
plt.xticks(rotation=45)
plt.title('Top 10 cheap states in USA by average price')
plt.xlabel('State')
plt.ylabel('Average Price')
plt.tight_layout()
fig2.savefig('plot2.png', dpi=300)



# Top 10 cheap regions by average price in California
# Calculate the average price for each region in california
state_filtered_df = df.filter(df['state'] == 'ca')
avg_price_ca_rdd = (
    state_filtered_df
    .groupBy('region_first')
    .agg({'price': 'mean'})
    .rdd
    .map(lambda row: (row['region_first'], row['avg(price)']))
    .sortBy(lambda x: x[1], ascending=True)
)

# Take the top 10 regions
top_10_states = avg_price_ca_rdd.take(10)

# Collect data to the driver for visualization
regions_ca, avg_prices = zip(*top_10_states)

# Create a bar chart
fig3 = plt.figure(figsize=(15, 8))
plt.bar(regions_ca, avg_prices)
plt.xticks(rotation=45)
plt.title('Top 10 cheap regions in California by average price')
plt.xlabel('Regions')
plt.ylabel('Average Price')
plt.tight_layout()
fig3.savefig('plot3.png', dpi=300)


# Top 10 cheap regions by average price in Florida
# Calculate the average price for each region of florida
state_filtered_df = df.filter(df['state'] == 'fl')
avg_price_fl_rdd = (
    state_filtered_df
    .groupBy('region_first')
    .agg({'price': 'mean'})
    .rdd
    .map(lambda row: (row['region_first'], row['avg(price)']))
    .sortBy(lambda x: x[1], ascending=True)
)

# Take the top 10 regions
top_10_states = avg_price_fl_rdd.take(10)

# Collect data to the driver for visualization
regions_fl, avg_prices = zip(*top_10_states)

# Create a bar chart
fig4 = plt.figure(figsize=(15, 8))
plt.bar(regions_fl, avg_prices)
plt.xticks(rotation=45)
plt.title('Top 10 cheap regions in Florida by average price')
plt.xlabel('Regions')
plt.ylabel('Average Price')
plt.tight_layout()
fig4.savefig('plot4.png', dpi=300)


#Count of Apartment Types in Each Region of California
state_filtered_df = df.filter(df['state'] == 'ca')

# Group by 'region_first' within the specified state and calculate counts for each 'type'
df_grouped_rdd = (
    state_filtered_df
    .groupBy('region_first')
    .agg(
        sum(when(state_filtered_df['type'] == 'apartment', 1).otherwise(0)).alias('apartment_count'),
        sum(when(state_filtered_df['type'] == 'condo', 1).otherwise(0)).alias('condo_count'),
        sum(when(state_filtered_df['type'] == 'duplex', 1).otherwise(0)).alias('duplex_count'),
        sum(when(state_filtered_df['type'] == 'house', 1).otherwise(0)).alias('house_count')
    )
    .rdd
    .map(lambda row: (row['region_first'], row['apartment_count'], row['condo_count'], row['duplex_count'], row['house_count']))
)


# Collect data to the driver for plotting
data_for_plot = df_grouped_rdd.collect()

regions = [row[0] for row in data_for_plot]
regions = [row[0] for row in data_for_plot]
apartment_counts = [row[1] for row in data_for_plot]  # Using index 1 for 'apartment_count'
condo_counts = [row[2] for row in data_for_plot]      # Using index 2 for 'condo_count'
duplex_counts = [row[3] for row in data_for_plot]     # Using index 3 for 'duplex_count'
house_counts = [row[4] for row in data_for_plot]      # Using index 4 for 'house_count'

# Set up the figure and axes
fig5, ax = plt.subplots(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions for each bar
bar_positions = np.arange(len(regions))

# Plot each type side by side
ax.bar(bar_positions - 1.5 * bar_width, apartment_counts, width=bar_width, label='apartment')
ax.bar(bar_positions - 0.5 * bar_width, condo_counts, width=bar_width, label='condo')
ax.bar(bar_positions + 0.5 * bar_width, duplex_counts, width=bar_width, label='duplex')
ax.bar(bar_positions + 1.5 * bar_width, house_counts, width=bar_width, label='house')

# Add text labels on top of each bar
for i, value in enumerate(apartment_counts + condo_counts + duplex_counts + house_counts):
    ax.text(bar_positions[i % len(bar_positions)] + (i // len(bar_positions) - 1.5) * bar_width, value + 0.1, str(value), ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('Region')
ax.set_ylabel('Count')
ax.set_title('Count of Apartment Types in Each Region of California')

# Set the x-axis ticks to the middle of the grouped bars
ax.set_xticks(bar_positions)
ax.set_xticklabels(regions)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')

# Show the legend
ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
fig5.savefig('plot5.png', dpi=300)


# Count of Apartment Types in Each State of USA
df_grouped_rdd = (
    df.groupBy("state")
    .agg(
        sum(when(state_filtered_df['type'] == 'apartment', 1).otherwise(0)).alias('apartment_count'),
        sum(when(state_filtered_df['type'] == 'condo', 1).otherwise(0)).alias('condo_count'),
        sum(when(state_filtered_df['type'] == 'duplex', 1).otherwise(0)).alias('duplex_count'),
        sum(when(state_filtered_df['type'] == 'house', 1).otherwise(0)).alias('house_count')
    )
    .rdd
    .map(lambda row: (row['state'], row['apartment_count'], row['condo_count'], row['duplex_count'], row['house_count']))
)


# Collect data to the driver for plotting
data_for_plot = df_grouped_rdd.collect()

states = [row[0] for row in data_for_plot]
apartment_counts = [row[1] for row in data_for_plot]  # Using index 1 for 'apartment_count'
condo_counts = [row[2] for row in data_for_plot]      # Using index 2 for 'condo_count'
duplex_counts = [row[3] for row in data_for_plot]     # Using index 3 for 'duplex_count'
house_counts = [row[4] for row in data_for_plot]      # Using index 4 for 'house_count'

# Set up the figure and axes
fig6, ax = plt.subplots(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions for each bar
bar_positions = np.arange(len(states))

# Plot each type side by side
ax.bar(bar_positions - 1.5 * bar_width, apartment_counts, width=bar_width, label='apartment')
ax.bar(bar_positions - 0.5 * bar_width, condo_counts, width=bar_width, label='condo')
ax.bar(bar_positions + 0.5 * bar_width, duplex_counts, width=bar_width, label='duplex')
ax.bar(bar_positions + 1.5 * bar_width, house_counts, width=bar_width, label='house')

# Add text labels on top of each bar
for i, value in enumerate(apartment_counts + condo_counts + duplex_counts + house_counts):
    ax.text(bar_positions[i % len(bar_positions)] + (i // len(bar_positions) - 1.5) * bar_width, value + 0.1, str(value), ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('Region')
ax.set_ylabel('Count')
ax.set_title('Count of Apartment Types in Each State of USA')

# Set the x-axis ticks to the middle of the grouped bars
ax.set_xticks(bar_positions)
ax.set_xticklabels(states)

# Rotate x-axis labels by 45 degrees
# plt.xticks(rotation=45, ha='right')

# Show the legend
ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
fig6.savefig('plot6.png', dpi=300)



# Top 10 states in USA with large average sqfeet housing
# Perform grouping and sorting operations
df_grouped_rdd = (
    df.groupBy('state')
    .agg(mean('sqfeet').alias('avg_sqfeet'))
    .rdd
    .sortBy(lambda x: x[1], ascending=False)  # Sort by average sqfeet in descending order
)

# Take the top 10 states
top_10_states = df_grouped_rdd.take(10)

# Collect data to the driver for further processing or visualization
states, avg_sqfeet = zip(*top_10_states)

states = [row[0] for row in top_10_states]
avg_sqfeet = [row[1] for row in top_10_states]

# Set up the figure and axes
fig7, ax = plt.subplots(figsize=(15, 8))

# Create a bar chart
plt.bar(states, avg_sqfeet)

# Set labels and title
plt.title('Top 10 states in USA with large average sqfeet housing')
plt.xlabel('State')
plt.ylabel('Average Sqfeet')
plt.tight_layout() 

fig7.savefig('plot7.png', dpi=300)



# Top 10 regions in California with large average sqfeet housing
# Perform filtering and grouping operations
state_filtered_df = df.filter(df['state'] == 'ca')

# Group by 'region_first' within the specified state and calculate the mean of 'sqfeet'
df_grouped_rdd = (
    state_filtered_df
    .groupBy('region_first')
    .agg(mean('sqfeet').alias('avg_sqfeet'))
    .rdd
)

# Take the top 10 regions based on average sqfeet
df_top_rdd = (
    df_grouped_rdd
    .sortBy(lambda x: x[1], ascending=False)  # Sort by average sqfeet in descending order
    .take(10)
)

regions = [row[0] for row in df_top_rdd]
avg_sqfeet = [row[1] for row in df_top_rdd]

# Set up the figure and axes
fig8, ax = plt.subplots(figsize=(15, 8))

# Create a bar chart
plt.bar(regions, avg_sqfeet)
plt.xticks(rotation=45)

# Set labels and title
plt.title('Top 10 regions in California with large average sqfeet housing')
plt.xlabel('Region')
plt.ylabel('Average Sqfeet')
fig8.savefig('plot8.png', dpi=300)



# Count of Bedroom types in Housing in Each Region of California
# Perform filtering and grouping operations
state_filtered_df = df.filter(df['state'] == 'ca')

# Group by 'region_first' within the specified state and calculate counts for each 'beds' value
df_grouped_rdd = (
    state_filtered_df
    .groupBy('region_first')
    .agg(
        sum(when(state_filtered_df['beds'] == '0', 1).otherwise(0)).alias('0b'),
        sum(when(state_filtered_df['beds'] == '1', 1).otherwise(0)).alias('1b'),
        sum(when(state_filtered_df['beds'] == '2', 1).otherwise(0)).alias('2b'),
        sum(when(state_filtered_df['beds'] == '3', 1).otherwise(0)).alias('3b'),
        sum(when(state_filtered_df['beds'] == '4', 1).otherwise(0)).alias('4b')
    )
    .rdd
)


# Collect data to the driver for plotting
data_for_plot = df_grouped_rdd.collect()

regions = [row[0] for row in data_for_plot]
beds_0_counts = [row[1] for row in data_for_plot]
beds_1_counts = [row[2] for row in data_for_plot]
beds_2_counts = [row[3] for row in data_for_plot]
beds_3_counts = [row[4] for row in data_for_plot]
beds_4_counts = [row[5] for row in data_for_plot]

# Set up the figure and axes
fig9, ax = plt.subplots(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions for each bar
bar_positions = np.arange(len(regions))

# Plot each type side by side
ax.bar(bar_positions - 2 * bar_width, beds_0_counts, width=bar_width, label='0b')
ax.bar(bar_positions - bar_width, beds_1_counts, width=bar_width, label='1b')
ax.bar(bar_positions, beds_2_counts, width=bar_width, label='2b')
ax.bar(bar_positions + bar_width, beds_3_counts, width=bar_width, label='3b')
ax.bar(bar_positions + 2 * bar_width, beds_4_counts, width=bar_width, label='4b')

# Add text labels on top of each bar
for i, value in enumerate(beds_0_counts + beds_1_counts + beds_2_counts + beds_3_counts + beds_4_counts):
    ax.text(bar_positions[i % len(bar_positions)] + (i // len(bar_positions) - 2) * bar_width, value + 0.1, str(value), ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('Bedroom Type')
ax.set_ylabel('Count')
ax.set_title('Count of Bedroom types in Housing in Each Region of California')

# Set the x-axis ticks to the middle of the grouped bars
ax.set_xticks(bar_positions)
ax.set_xticklabels(regions)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')

# Show the legend
ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
fig9.savefig('plot9.png', dpi=300)



# Count of Bedroom types in Housing in Each State of USA
# Group by 'state' within the united states
df_grouped_rdd = (
    df
    .groupBy('state')
    .agg(
        sum(when(df['beds'] == '0', 1).otherwise(0)).alias('0b'),
        sum(when(df['beds'] == '1', 1).otherwise(0)).alias('1b'),
        sum(when(df['beds'] == '2', 1).otherwise(0)).alias('2b'),
        sum(when(df['beds'] == '3', 1).otherwise(0)).alias('3b'),
        sum(when(df['beds'] == '4', 1).otherwise(0)).alias('4b')
    )
    .rdd
)

states = [row[0] for row in data_for_plot]
beds_0_counts = [row[1] for row in data_for_plot]
beds_1_counts = [row[2] for row in data_for_plot]
beds_2_counts = [row[3] for row in data_for_plot]
beds_3_counts = [row[4] for row in data_for_plot]
beds_4_counts = [row[5] for row in data_for_plot]

# Set up the figure and axes
fig10, ax = plt.subplots(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions for each bar
bar_positions = np.arange(len(states))

# Plot each type side by side
ax.bar(bar_positions - 2 * bar_width, beds_0_counts, width=bar_width, label='0b')
ax.bar(bar_positions - bar_width, beds_1_counts, width=bar_width, label='1b')
ax.bar(bar_positions, beds_2_counts, width=bar_width, label='2b')
ax.bar(bar_positions + bar_width, beds_3_counts, width=bar_width, label='3b')
ax.bar(bar_positions + 2 * bar_width, beds_4_counts, width=bar_width, label='4b')

# Add text labels on top of each bar
for i, value in enumerate(beds_0_counts + beds_1_counts + beds_2_counts + beds_3_counts + beds_4_counts):
    ax.text(bar_positions[i % len(bar_positions)] + (i // len(bar_positions) - 2) * bar_width, value + 0.1, str(value), ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('State')
ax.set_ylabel('Count')
ax.set_title('Count of Bedroom types in Housing in Each State')

# Set the x-axis ticks to the middle of the grouped bars
ax.set_xticks(bar_positions)
ax.set_xticklabels(states)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')

# Show the legend
ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
fig10.savefig('plot10.png', dpi=300)



# Count of Housing That Allows Pets in Each State
# Perform grouping operations
df_grouped_rdd = (
    df
    .groupBy('state')
    .agg(
        sum(when(df['cats_allowed'] == '0', 1).otherwise(0)).alias('cats_allowed'),
        sum(when(df['dogs_allowed'] == '1', 1).otherwise(0)).alias('dogs_allowed')
    )
    .rdd
)


# Collect data to the driver for plotting
data_for_plot = df_grouped_rdd.collect()

# Convert the collected data to NumPy arrays for plotting
states = [row[0] for row in data_for_plot]
cats_allowed_counts = [row[1] for row in data_for_plot]
dogs_allowed_counts = [row[2] for row in data_for_plot]

# Set up the figure and axes
fig11, ax = plt.subplots(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions for each bar
bar_positions = np.arange(len(states))

# Plot each type side by side
ax.bar(bar_positions - bar_width, cats_allowed_counts, width=bar_width, label='Cats Allowed')
ax.bar(bar_positions, dogs_allowed_counts, width=bar_width, label='Dogs Allowed')

# Add text labels on top of each bar
for i, value in enumerate(cats_allowed_counts + dogs_allowed_counts):
    ax.text(bar_positions[i % len(bar_positions)] - bar_width / 2, value + 0.1, str(value), ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('State')
ax.set_ylabel('Count')
ax.set_title('Count of Housing That Allows Pets in Each State')

# Set the x-axis ticks to the middle of the grouped bars
ax.set_xticks(bar_positions)
ax.set_xticklabels(states)

# Rotate x-axis labels by 45 degrees
# plt.xticks(rotation=45, ha='right')

# Show the legend
ax.legend(title='Pet Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
fig11.savefig('plot11.png', dpi=300)



# Count of Housing That Allows Pets in Regions of California
# Filter the DataFrame for a specific state (e.g., 'California')
state_df = df.filter(df['state'] == 'ca')

# Group by 'region_first' within the specified state and calculate the mean of 'price'
df_grouped_rdd = (
    state_df
    .groupBy('region_first')
    .agg(
        sum(when(state_df['cats_allowed'] == '0', 1).otherwise(0)).alias('cats_allowed'),
        sum(when(state_df['dogs_allowed'] == '1', 1).otherwise(0)).alias('dogs_allowed')
    )
    .rdd
)

# Collect data to the driver for plotting
data_for_plot = df_grouped_rdd.collect()

# Convert the collected data to NumPy arrays for plotting
regions = [row[0] for row in data_for_plot]
cats_allowed_counts = [row[1] for row in data_for_plot]
dogs_allowed_counts = [row[2] for row in data_for_plot]

# Set up the figure and axes
fig12, ax = plt.subplots(figsize=(12, 8))

# Set the width of the bars
bar_width = 0.2

# Set the positions for each bar
bar_positions = np.arange(len(regions))

# Plot each type side by side
ax.bar(bar_positions - bar_width, cats_allowed_counts, width=bar_width, label='Cats Allowed')
ax.bar(bar_positions, dogs_allowed_counts, width=bar_width, label='Dogs Allowed')

# Add text labels on top of each bar
for i, value in enumerate(cats_allowed_counts + dogs_allowed_counts):
    ax.text(bar_positions[i % len(bar_positions)] - bar_width / 2, value + 0.1, str(value), ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('Regions')
ax.set_ylabel('Count')
ax.set_title('Count of Housing That Allows Pets in Regions of California')

# Set the x-axis ticks to the middle of the grouped bars
ax.set_xticks(bar_positions)
ax.set_xticklabels(regions)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')

# Show the legend
ax.legend(title='Pet Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
fig12.savefig('plot12.png', dpi=300)

# Code to save all the figures of Analysis on S3
for filename in filenames:
    with open(filename, 'rb') as file:
        s3.upload_fileobj(file, 'usa-housing-price', filename)