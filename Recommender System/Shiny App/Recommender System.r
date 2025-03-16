library(recommenderlab)
library(dplyr)
library(tm)

# Load the data
ratings <- read.csv(
  "C:/Users/Msi/Desktop/R folder/Group Work/Recommender System/ml-latest-small/ml-latest-small/ratings.csv"
)
ratings_matrix <- as(ratings, "realRatingMatrix")

# Clean the data (keep users/movies with at least 5 ratings)
min_ratings <- 5
ratings_matrix <- ratings_matrix[
  rowCounts(ratings_matrix) >= min_ratings,
  colCounts(ratings_matrix) >= min_ratings
]


# Normalize the data
ratings_matrix <- normalize(ratings_matrix)

# Split the data (specify `given` and `goodRating`)
set.seed(123)
train_test <- evaluationScheme(
  ratings_matrix,
  method = "split",
  train = 0.8,
  given = 5,
  goodRating = 4
)

# Extract datasets
train <- getData(train_test, "train")
test <- getData(train_test, "known")
test_unknown <- getData(train_test, "unknown")

# Train Collaborative Filtering (User-Based)
recommender_ubcf <- Recommender(train, method = "UBCF")
predictions_ubcf <- predict(recommender_ubcf, test_unknown, type = "ratings")
accuracy_ubcf <- calcPredictionAccuracy(predictions_ubcf, test, given = 5, goodRating = 4)

# Train Collaborative Filtering (Item-Based)
recommender_ibcf <- Recommender(train, method = "IBCF")
predictions_ibcf <- predict(recommender_ibcf, test_unknown, type = "ratings")
accuracy_ibcf <- calcPredictionAccuracy(predictions_ibcf, test, given = 5, goodRating = 4)

# Save models
saveRDS(recommender_ubcf, "UBCF_model.rds")
saveRDS(recommender_ibcf, "IBCF_model.rds")

# Content-Based Filtering (requires movie metadata)
movies <- read.csv(
  "C:/Users/Msi/Desktop/R folder/Group Work/Recommender System/ml-latest-small/ml-latest-small/movies.csv"
)

# Create a Document-Term Matrix from genres
corpus <- Corpus(VectorSource(movies$genres))
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))

# Convert DTM to a binary matrix (presence/absence of terms)
dtm_binary <- apply(as.matrix(dtm), 2, function(x) as.numeric(x > 0))

# Convert binary matrix to a realRatingMatrix
content_ratings_matrix <- as(dtm_binary, "realRatingMatrix")

# Train Content-Based Filtering using Cosine similarity
content_model <- Recommender(content_ratings_matrix, method = "UBCF")
saveRDS(content_model, "Content_model.rds")

