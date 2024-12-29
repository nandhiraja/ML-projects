# Movie Recommendation System 
### Check out the live demo here: [Movie Recommendation System](https://movie-recommendation-system-nr.streamlit.app/)


This is a simple **Movie Recommendation System** built using Python. The application leverages the IMDb movie dataset to recommend movies based on user input and preferences.

## Features
- Recommends movies based on user-input movie names.
- Displays a selection of random movie names for exploration.
- Uses **TF-IDF Vectorizer** and **cosine similarity** to find close matches and provide recommendations.

## Requirements
- Python 3.8 or above
- Libraries:
  - numpy
  - pandas
  - streamlit
  - sklearn
  - openpyxl

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the datasets (`cleaned_dataset.csv` and `recom_dataset.csv`) are in the project folder.

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Follow the on-screen instructions:
   - Choose your favorite movie genre.
   - View random movie suggestions.
   - Enter a movie name to get personalized recommendations.

3. Exit the app using the **Exit** button when done.

## Example
### User Interface Preview
(Add screenshots of the app interface here)

### Output Example
- **Input**: Movie name entered by the user.
- **Output**: List of 10 recommended movies ranked by similarity.



---

Enjoy exploring and finding your next favorite movie! 🎥
