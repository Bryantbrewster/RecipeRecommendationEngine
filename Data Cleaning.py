from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType


spark = SparkSession.builder.getOrCreate()
print(spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())


# Path to your CSV file
csv_file_path = 'datasets/RAW_interactions.csv'

# Read the CSV file into a DataFrame
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Show the first few rows of the DataFrame
df.show(40)

data_types = df.dtypes
print(data_types)
df = df.drop('review', 'date')
df.show(40)

# Filter out rows where 'user_id' is not numeric
# This regular expression matches any string that contains a non-digit character
filtered_df = df.filter(
    ~F.col("user_id").rlike("[^0-9]") &
    ~F.col("recipe_id").rlike("[^0-9]") &
    ~F.col("date").rlike("[^0-9]")
)
filtered_df.show(40)

contains_text = filtered_df.filter(F.col("rating").rlike("[^0-9]+"))
any_text_rows = contains_text.count() > 0
contains_text.show()
print("Are there any rows with text in 'your_column_name'? ", "Yes" if any_text_rows else "No")

# Convert 'user_id' and 'recipe_id' to IntegerType, and 'rating' to FloatType
filtered_df = df.withColumn("user_id", col("user_id").cast(IntegerType())) \
       .withColumn("recipe_id", col("recipe_id").cast(IntegerType())) \
       .withColumn("rating", col("rating").cast(FloatType()))

no_zero = filtered_df.dropna(how='any')
just_ratings_df = no_zero.filter(no_zero['rating'] != 0.0)
# Show the resulting DataFrame to verify the changes
just_ratings_df.show()
just_ratings_df.printSchema()
# To confirm the data types have been changed, you can print the schema
just_ratings_df.printSchema()

output_path = 'datasets/interactions'
just_ratings_df.write.parquet(output_path, mode="overwrite")
