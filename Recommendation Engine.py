import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, count, lit, explode, isnull
from pyspark.sql.types import DoubleType, FloatType, IntegerType, StructType, StructField
import warnings
import os

warnings.filterwarnings("ignore")


# Initialize SparkSession
spark = SparkSession.builder \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.initialExecutors", "2") \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "10") \
    .config("spark.dynamicAllocation.executorIdleTimeout", "60s") \
    .config("spark.dynamicAllocation.shuffleTracking.enabled", "true") \
    .config("spark.shuffle.service.enabled", "true") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("checkpoints")


user_id = 7
os.system("cls")
print()
print("Welcome to the recipe recommender! Providing recipe ideas so you don't have to even think about what to make for dinner.")
print()
new_vs_returning = input("Are you a new or returning user? (new/returning) ")
# print(“\n” * 1000)
os.system("cls")
print()
if new_vs_returning.lower() == 'new':

    # Read datasets and initialize dataframe
    path_to_onboarding_recipes = "datasets/master-tables/Onboarding_recipes.csv"
    onboarding_recipes_df = spark.read.csv(path_to_onboarding_recipes, header=True, inferSchema=True)


    cuisine_mapping = {
        '1': 'american',
        '2': 'asian',
        '3': 'italian',
        '4': 'mexican',
        '5': 'barbecue',
        '6': 'indian',
        '7': 'african',
        '8': 'greek',
        '9': 'french',
        '10': 'german',
        '11': 'thai',
        '12': 'middle-eastern'
    }
    print(f"Welcome! I can't wait to get started. Your user id is {user_id}! \nRemember this ID number, you'll need it to access your personalized recommendations in the future.")
    print("Before we get to your recommendations, let's gather some information about what you like to cook!")
    print()
    print()
    print("Question 1: How long are you willing to spend cooking?")
    print("1. 30 minutes or less")
    print("2. 60 minutes or less")
    print("3. Doesn't matter to me as long as it tastes good!")
    cook_time = int(input("Select your answer (1, 2, or 3) "))
    os.system("cls")
    print()
    print()
    print("Question 2: What are your favorite types of cuisine?")
    print('1. American')
    print('2. Asian')
    print('3. Italian')
    print('4. Mexican')
    print('5. Barbecue')
    print('6. Indian')
    print('7. African')
    print('8. Greek')
    print('9. French')
    print('10. German')
    print('11. Thai')
    print('12. Middle Eastern')
    cuisine_pref = input("Enter all that apply, separated by commas (e.g., 1,4,6) ")
    os.system("cls")
    cuisine_choices = cuisine_pref.split(',')
    favorite_cuisines = [cuisine_mapping[number.strip()] for number in cuisine_choices if
                         number.strip() in cuisine_mapping]
    print()
    print()
    print("Question 3: Final question! Allow spicy foods to be included in your suggestion mix? ")
    print('1. Heck yeah!')
    print('2. No way!')
    spicy_pref = int(input("Enter your answer (1 or 2) "))
    os.system("cls")
    print()
    print()
    print("Perfect! Now, let's get you started with some initial recommendations!")
    print("Loading delicious recipes...")
    print("Loading delicious recipes...")
    print("Loading delicious recipes...")
    print()
    print()
    favorite_cuisine_onboarding_recipes = onboarding_recipes_df[onboarding_recipes_df['Cuisine'].isin(favorite_cuisines)]
    # favorite_cuisine_onboarding_recipes.show()

    simulated_reviews_df = favorite_cuisine_onboarding_recipes.select(
        lit(user_id).alias("user_id"),  # Creates a column with the user_id for all rows
        "recipe_id",  # Keeps the recipe_id from the filtered DataFrame
        col("Sum of average_rating").cast(FloatType()).alias("rating")  # Renames the average_rating column to 'rating'
    )




    # Append the new DataFrame initial onboarding rows onto the master interaction dataset
    simulated_reviews_df.write.mode('append').parquet('datasets/interactions')

    # Read datasets and initialize dataframe
    path_to_directory = "datasets/interactions"
    interactions_df = spark.read.parquet(path_to_directory)

    '''Testing area for filtered versions of the dataframe for training'''
    # Group by 'recipe_id' and count the number of reviews
    recipe_review_counts = interactions_df.groupBy("recipe_id").agg(count("rating").alias("num_reviews"))

    # Filter for recipes with 5 or more reviews
    popular_recipes = recipe_review_counts.filter(col("num_reviews") >= 5)

    # Join the filtered DataFrame with the original interactions DataFrame to get only interactions for popular recipes
    filtered_interactions_df = interactions_df.join(popular_recipes, "recipe_id").select(interactions_df["*"])
    num_popular_recipes = popular_recipes.count()


    # Now, 'filtered_interactions_df' contains only the interactions for recipes with 5 or more reviews
    '''Fitting a basic model'''
    '''Split our data into train/test'''
    (training_data, test_data) = filtered_interactions_df.randomSplit([0.8, 0.2])

    '''Build the ALS model'''
    als = ALS(userCol="user_id", itemCol="recipe_id", ratingCol="rating",
              rank=5, maxIter=20, regParam=0.15, nonnegative=True,
              coldStartStrategy="drop", implicitPrefs=False)

    '''Fit the model'''
    model = als.fit(training_data)
    recommendations = model.recommendForAllUsers(50)
    # Create a DataFrame containing just the specified user_id

    exploded_recs = recommendations.select(recommendations.user_id,
                                           explode(recommendations.recommendations).alias("recommendation"))

    # Selecting the user_id, recipe_id, and rating from the exploded recommendations
    recommendation_details = exploded_recs.select(
        "user_id",
        col("recommendation.recipe_id").alias("recipe_id"),
        col("recommendation.rating").alias("rating")
    )

    # Filter the DataFrame for the specific user
    specific_user_recommendations = recommendation_details.filter(recommendation_details.user_id == user_id)
    sorted_recs = specific_user_recommendations.orderBy("rating", ascending=False)
    # Read in master recipe file
    master_recipe_df = spark.read.csv("datasets/master-tables/recipes_master.csv", header=True, inferSchema=True)
    # filtered_csv_df = master_recipe_df.filter((master_recipe_df.time_to_cook == '15-minutes-or-less') |
    #                                           (master_recipe_df.time_to_cook == '30-minutes-or-less'))

    joined_df = sorted_recs.join(master_recipe_df, "recipe_id")
    # Select columns I want to show
    final_rec_df = joined_df.select(
        "user_id",
        "recipe_id",
        "rating",
        "name",
        "recipe_url",
        "time_to_cook",
        "spicy_or_no"
    )

    # Filter rows for time_to_cook user preference
    if cook_time == 1:
        filtered_rec_df = final_rec_df.filter(final_rec_df.time_to_cook == '30-minutes-or-less')
        if spicy_pref == 2:
            filtered_rec_df = filtered_rec_df.filter(filtered_rec_df.spicy_or_no == 'not spicy')
    elif cook_time == 2:
        filtered_rec_df = final_rec_df.filter((final_rec_df.time_to_cook == '30-minutes-or-less') |
                                              (final_rec_df.time_to_cook == '60-minutes-or-less'))
        if spicy_pref == 2:
            filtered_rec_df = filtered_rec_df.filter(filtered_rec_df.spicy_or_no == 'not spicy')
    else:
        filtered_rec_df = final_rec_df
        if spicy_pref == 2:
            filtered_rec_df = filtered_rec_df.filter(filtered_rec_df.spicy_or_no == 'not spicy')


    print('Here are some initial recipe recommendations based on recipes popular with others like you!')
    print('Be sure to come back and review recipes to improve your personalized recommendations!')
    filtered_rec_df.orderBy(final_rec_df.rating.desc()).show(truncate=False)



if new_vs_returning.lower() == 'returning':
    print('Welcome back!')
    print()
    print("What's your user id?")
    user_id = int(input('Enter user id: '))
    print()
    print('Thanks! Do you have any recipes you want to leave a review for? (Yes/No)')
    has_recipe_to_review = input()
    if has_recipe_to_review.lower() == 'yes':
        recipe_id_to_leave_rating_on = int(input('Enter the recipe id of the recipe you want to review: '))
        print()
        print('What rating (1-5) would you rate this recipe? Decimals are okay.')
        rating_to_give_recipe = float(input())
        print()
        os.system("cls")
        # define schema
        schema = StructType([
            StructField("user_id", IntegerType(), True),
            StructField("recipe_id", IntegerType(), True),
            StructField("rating", FloatType(), True)
        ])
        # create new single-row dataframe
        new_review = spark.createDataFrame(
            [(user_id, recipe_id_to_leave_rating_on, rating_to_give_recipe)],
            schema=schema
        )
        # append to interactions parquet file as a new review by the user
        new_review.write.mode("append").parquet("datasets/interactions")
        print(f'Your rating has been recorded for recipe {recipe_id_to_leave_rating_on}')
    print()
    print("Sounds great. Let's get you some recipes to try!")
    print("First, how long are you willing to spend cooking this time?")
    print("1. 30 minutes or less")
    print("2. 60 minutes or less")
    print("3. Doesn't matter to me as long as it tastes good!")
    cook_time = int(input("Select your answer (1, 2, or 3) "))
    print()
    print()
    print("Allow spicy foods to be included in your suggestion mix? ")
    print("1. Heck yeah!")
    print('2. No way!')
    spicy_pref = int(input("Enter your answer (1 or 2) "))
    print()
    os.system("cls")
    print("Great! Now let's whip up some recommendations!")
    print("Retrieving your preferences...")
    print("Loading delicious recipes...")
    print("Loading delicious recipes...")
    print("Loading delicious recipes...")
    print()

    # Read datasets and initialize dataframe
    path_to_directory = "datasets/interactions"
    interactions_df = spark.read.parquet(path_to_directory)


    '''Testing area for filtered versions of the dataframe for training'''
    # Group by 'recipe_id' and count the number of reviews
    recipe_review_counts = interactions_df.groupBy("recipe_id").agg(count("rating").alias("num_reviews"))

    # Filter for recipes with 5 or more reviews
    popular_recipes = recipe_review_counts.filter(col("num_reviews") >= 5)

    # Join the filtered DataFrame with the original interactions DataFrame to get only interactions for popular recipes
    filtered_interactions_df = interactions_df.join(popular_recipes, "recipe_id").select(interactions_df["*"])
    num_popular_recipes = popular_recipes.count()


    # Now, 'filtered_interactions_df' contains only the interactions for recipes with 5 or more reviews
    '''Fitting a basic model'''
    '''Split our data into train/test'''
    (training_data, test_data) = filtered_interactions_df.randomSplit([0.8, 0.2])

    '''Build the ALS model'''
    als = ALS(userCol="user_id", itemCol="recipe_id", ratingCol="rating",
              rank=5, maxIter=20, regParam=0.15, nonnegative=True,
              coldStartStrategy="drop", implicitPrefs=False)

    '''Fit the model'''
    model = als.fit(training_data)
    recommendations = model.recommendForAllUsers(20)
    # Create a DataFrame containing just the specified user_id

    exploded_recs = recommendations.select(recommendations.user_id,
                                           explode(recommendations.recommendations).alias("recommendation"))

    # Selecting the user_id, recipe_id, and rating from the exploded recommendations
    recommendation_details = exploded_recs.select(
        "user_id",
        col("recommendation.recipe_id").alias("recipe_id"),
        col("recommendation.rating").alias("rating")
    )


    # Filter the DataFrame for the specific user
    specific_user_recommendations = recommendation_details.filter(recommendation_details.user_id == user_id)
    sorted_recs = specific_user_recommendations.orderBy("rating", ascending=False)
    # Read in master recipe file
    master_recipe_df = spark.read.csv("datasets/master-tables/recipes_master.csv", header=True, inferSchema=True)

    joined_df = sorted_recs.join(master_recipe_df, "recipe_id")
    # Select columns I want to show
    final_rec_df = joined_df.select(
        "user_id",
        "recipe_id",
        "rating",
        "name",
        "recipe_url",
        "time_to_cook",
        "spicy_or_no"
    )

    # Filter rows for time_to_cook user preference
    if cook_time == 1:
        filtered_rec_df = final_rec_df.filter(final_rec_df.time_to_cook == '30-minutes-or-less')
        if spicy_pref == 2:
            filtered_rec_df = filtered_rec_df.filter(filtered_rec_df.spicy_or_no == 'not spicy')
    elif cook_time == 2:
        filtered_rec_df = final_rec_df.filter((final_rec_df.time_to_cook == '30-minutes-or-less') |
                                              (final_rec_df.time_to_cook == '60-minutes-or-less'))
        if spicy_pref == 2:
            filtered_rec_df = filtered_rec_df.filter(filtered_rec_df.spicy_or_no == 'not spicy')
    else:
        filtered_rec_df = final_rec_df
        if spicy_pref == 2:
            filtered_rec_df = filtered_rec_df.filter(filtered_rec_df.spicy_or_no == 'not spicy')


    print('Whipping up your top recipe recommendations based on your preferences and others like you!')
    print("The more you come back and leave recipe ratings, the better your suggestions will get!")
    print()
    filtered_rec_df.orderBy(final_rec_df.rating.desc()).show(truncate=False)




spark.stop()







