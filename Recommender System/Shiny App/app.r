library(shiny)
library(recommenderlab)
library(dplyr)


# Load pre-trained models and data
UBCF_model <- readRDS("UBCF_model.rds")
IBCF_model <- readRDS("IBCF_model.rds")
Content_model <- readRDS("Content_model.rds")



# Load data for reference
ratings <- read.csv("ratings.csv")
movies <- read.csv("movies.csv")

# Convert ratings to realRatingMatrix
ratings_matrix <- as(ratings, "realRatingMatrix")

# UI
ui <- fluidPage(
  titlePanel("Movie Recommendation System"),
  
  sidebarLayout(
    sidebarPanel(
      h3("User-Based Collaborative Filtering"),
      numericInput("user_id", "Enter User ID:", value = 1, min = 1, max = max(ratings$userId)),
      actionButton("recommend_ubcf", "Get Recommendations (UBCF)"),
      
      h3("Item-Based Collaborative Filtering"),
      numericInput("movie_id", "Enter Movie ID:", value = 1, min = 1, max = max(movies$movieId)),
      actionButton("recommend_ibcf", "Get Recommendations (IBCF)"),
      
      h3("Content-Based Filtering"),
      textInput("movie_title", "Enter Movie Title:", value = ""),
      actionButton("recommend_content", "Get Recommendations (Content-Based)")
    ),
    
    mainPanel(
      h3("Recommendations"),
      tabsetPanel(
        tabPanel("UBCF", tableOutput("ubcf_recommendations")),
        tabPanel("IBCF", tableOutput("ibcf_recommendations")),
        tabPanel("Content-Based", tableOutput("content_recommendations"))
      )
    )
  )
)

# Align columns with the training data
align_columns <- function(new_matrix, training_matrix) {
  all_items <- colnames(training_matrix)
  new_matrix_data <- as(new_matrix, "matrix")  # Convert to standard matrix
  aligned_matrix <- matrix(0, nrow = nrow(new_matrix_data), ncol = length(all_items))
  colnames(aligned_matrix) <- all_items
  matching_columns <- intersect(colnames(new_matrix_data), all_items)
  aligned_matrix[, matching_columns] <- new_matrix_data[, matching_columns, drop = FALSE]
  as(aligned_matrix, "realRatingMatrix")  # Convert back to realRatingMatrix
}

# Server
server <- function(input, output, session) {
  
  # UBCF Recommendations
observeEvent(input$recommend_ubcf, {
  user_id <- input$user_id
  
  # Extract user ratings and align with the training matrix
  user_ratings <- ratings_matrix[user_id, , drop = FALSE]
  aligned_ratings <- align_columns(user_ratings, ratings_matrix)
  
  # Predict top-N recommendations for the user
  predictions <- predict(UBCF_model, aligned_ratings, type = "topNList", n = 5)
  recommendations <- as.data.frame(as(predictions, "list")[[1]])
  colnames(recommendations) <- "MovieID"
  
  # Map Movie IDs to Movie Titles
  recommendations <- recommendations %>%
    left_join(movies, by = c("MovieID" = "movieId"))
  
  output$ubcf_recommendations <- renderTable({
    recommendations[, c("MovieID", "title")]
  })
})
  
  # IBCF Recommendations
  observeEvent(input$recommend_ibcf, {
  movie_id <- input$movie_id
  
  # Extract movie ratings and align with the training matrix
  movie_ratings <- ratings_matrix[, movie_id, drop = FALSE]
  aligned_ratings <- align_columns(movie_ratings, ratings_matrix)
  
  # Predict top-N recommendations for the movie
  predictions <- predict(IBCF_model, aligned_ratings, type = "topNList", n = 5)
  recommendations <- as.data.frame(as(predictions, "list")[[1]])
  colnames(recommendations) <- "MovieID"
  
  # Map Movie IDs to Movie Titles
  recommendations <- recommendations %>%
    left_join(movies, by = c("MovieID" = "movieId"))
  
  output$ibcf_recommendations <- renderTable({
    recommendations[, c("MovieID", "title")]
  })
})
  
  # Content-Based Recommendations
  observeEvent(input$recommend_content, {
  movie_title <- input$movie_title
  
  # Find the movie ID based on the title
  movie_id <- movies %>%
    filter(title == movie_title) %>%
    pull(movieId)
  
  # Debugging: Print the movie ID
  print(paste("Movie ID:", movie_id))
  
  # Debugging: Print row names of content_ratings_matrix
  print(content_ratings_matrix)
  print("Row names of content_ratings_matrix:")
  print(rownames(content_ratings_matrix))
  
  if (length(movie_id) == 0) {
    output$content_recommendations <- renderTable({
      data.frame(Message = "No matching movie found.")
    })
  } else if (!as.character(movie_id) %in% rownames(content_ratings_matrix)) {
    # Handle case where movie_id is not in content_ratings_matrix
    output$content_recommendations <- renderTable({
      data.frame(Message = "Movie ID not found in content ratings matrix.")
    })
  } else {
    # Extract the genre vector for the selected movie
    movie_genres <- content_ratings_matrix[as.character(movie_id), , drop = FALSE]
    
    # Debugging: Print dimensions of movie_genres
    print(dim(movie_genres))  # Dimensions of the input matrix
    
    # Predict top-N recommendations for the movie
    predictions <- predict(Content_model, movie_genres, type = "topNList", n = 5)
    recommendations <- as.data.frame(as(predictions, "list")[[1]])
    colnames(recommendations) <- "MovieID"
    
    # Map Movie IDs to Movie Titles
    recommendations <- recommendations %>%
      left_join(movies, by = c("MovieID" = "movieId"))
    
    output$content_recommendations <- renderTable({
      recommendations[, c("MovieID", "title")]
    })
  }
})
}



shinyApp(ui,server )


