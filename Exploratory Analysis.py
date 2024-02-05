from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, avg, count

# Initialize SparkSession
spark = SparkSession.builder.getOrCreate()

# Read datasets and initialize dataframe
path_to_directory = "datasets/interactions"
df = spark.read.parquet(path_to_directory)
df.show(40)

number_of_reviews = df.count()
unique_recipes_reviewed = df.agg(countDistinct("recipe_id").alias("distinct_recipe_ids")).collect()[0]["distinct_recipe_ids"]
user_count = df.agg(countDistinct("user_id").alias("distinct_user_ids")).collect()[0]["distinct_user_ids"]
average_rating = df.agg(avg("rating").alias("average_rating")).collect()[0]["average_rating"]
# df.groupBy('recipe_id').count().orderBy('count', ascending=False).show()
# df.groupBy("user_id").count().agg(avg("count").alias("average_occurrences")).show()
# avg_ratings_per_user =
# avg_ratings_per_recipe =

ratings_count = df.groupBy("recipe_id").agg(count("rating").alias("ratings_count"))
# Calculate the average number of ratings per recipe
average_ratings_count = ratings_count.agg(avg("ratings_count").alias("average_ratings_count")).show()

print(f"Total number of reviews: {number_of_reviews}")
print(f"Unique Recipes: {unique_recipes_reviewed}")
print(f"Unique User Count: {user_count}")
print(f"Average recipe rating: {average_rating}")
print(f"Average reviews per recipe: {average_ratings_count}")

# Initialize SparkSession
spark = SparkSession.builder.appName("MedianReviewsPerRecipe").getOrCreate()

# Assuming 'df' is your DataFrame and 'recipe_id' is the column identifying recipes
# Step 1: Group by 'recipe_id' and count the number of reviews for each recipe
reviews_count = df.groupBy("recipe_id").agg(count("recipe_id").alias("reviews_count"))

# Step 2: Calculate the approximated median of the 'reviews_count' column
# The parameters are the column name, the quantiles as a list (0.5 for median), and the relative error (0 for exact, though it's an approximation)
median_reviews_approx = reviews_count.stat.approxQuantile("reviews_count", [0.5], 0.01)

print(f"Approximated Median Number of Reviews per Recipe: {median_reviews_approx[0]}")


# Show the result






# recipe_df = spark.read.csv("datasets/RAW_recipes.csv", header=True)
# recipe_df.filter(recipe_df["id"] == "27208").select("name").show()
# most_pop_recipe = spark.sql("SELECT name from recipe_view WHERE id = '27208' ").show()


spark.stop()