import pandas as pd  # Data handling
import numpy as np  # Numerical ops
from dotenv import load_dotenv  # Environment variables

from langchain.schema import Document  # LangChain document schema
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding model
from langchain_text_splitters import CharacterTextSplitter  # Text splitter
from langchain_chroma import Chroma  # Vector store

import gradio as gr # UI library

# Load environment variables (API tokens, config)
load_dotenv()

# ---------- Data Preparation ----------
# Read book metadata and prepare thumbnail URLs
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load pre-tagged descriptions for embedding
with open("tagged_description.txt", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Convert each line into a LangChain Document
raw_documents = [
    Document(page_content=line.strip())
    for line in content.split("\n")
    if line.strip()
]

# Split documents (no overlap in this case)
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=0,
    chunk_overlap=0
)
documents = text_splitter.split_documents(raw_documents)

# Create the Chroma vector store with Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)

# ---------- Recommendation Logic ----------
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Running a semantic similarity search, then filter and sort
    by category or emotional tone if specified.
    """
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    # Extract the ISBN13s from the top semantically-similar chunks
    books_list = [
        int(doc.page_content.strip('"').split()[0])
        for doc, _score in recs
    ]
    # Narrow down to actual book metadata
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)
    # Filter by category if specified
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    if tone == "Happy":
        book_recs = book_recs.sort_values("joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values("surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values("anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values("fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values("sadness", ascending=False)
    return book_recs

# ---------- Gradio UI ----------
def recommend_books(query: str, category: str, tone: str):
    """ Gradio UI logic """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..." if len(truncated_desc_split) > 30 else description
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    if not results:
        return [("cover-not-found.jpg", "No recommendations found. Please try a different query!")]
    return results

# Dropdown options for category and tone
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Random featured books
def get_featured_books():
    featured = books.sample(n=12) if len(books) >= 12 else books
    results = []
    for _, row in featured.iterrows():
        description = str(row.get("description", ""))
        description = " ".join(description.split()[:20]) + "..." if len(description.split()) > 20 else description
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        caption = f"{row['title']} by {authors_str}: {description}"
        results.append((row["large_thumbnail"], caption))
    return results

# ---------- UI Definition ----------

with gr.Blocks(
    theme=gr.themes.Glass(),
    css="""
/* --------------------------------------------------
   1. Gallery Item: Hover-Lift Card Effect
-------------------------------------------------- */
.gr-gallery-item {
  background: rgba(255,255,255,0.03);
  border-radius: 8px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.3);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.gr-gallery-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}

/* --------------------------------------------------
   2. Buttons: Modern Raised Look
-------------------------------------------------- */
.gradio-button {
  padding: 12px 24px !important;
  border-radius: 24px !important;
  font-weight: 600 !important;
  transition: box-shadow 0.2s;
}
.gradio-button:hover {
  box-shadow: 0 4px 12px rgba(0, 200, 255, 0.5);
}

/* --------------------------------------------------
   3. Gallery Entrance Animation
-------------------------------------------------- */
.gradio-gallery {
  opacity: 0;
  transform: translateY(20px);
  animation: fadeUp 0.5s forwards ease-out;
}
@keyframes fadeUp {
  to { opacity: 1; transform: translateY(0); }
}

/* --------------------------------------------------
   4. Smooth Scroll for Main Gallery
-------------------------------------------------- */
#gallery-box { scroll-behavior: smooth; }

/* --------------------------------------------------
   5. Featured Gallery Layout Tweaks
-------------------------------------------------- */
#featured-gallery {
  overflow-y: visible !important;
  overflow-x: hidden !important;
  padding-bottom: 8px;
  margin-bottom: 24px;
  height: auto !important;
}
#featured-gallery .gradio-gallery {
  display: flex !important;
  flex-wrap: wrap !important;
  min-width: unset !important;
  gap: 16px;
}
#featured-gallery .gradio-gallery img {
  width: 110px !important;
  height: 165px !important;
  object-fit: cover !important;
  border-radius: 6px;
}

/* --------------------------------------------------
   6. Footer ("About") Section
-------------------------------------------------- */
#project-info {
  margin-top: 64px !important;
  padding: 32px 24px !important;
  background: linear-gradient(135deg, #1e1f2a 0%, #2a2c3d 100%) !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
  position: relative !important;
  overflow: hidden !important;
  text-align: center !important;
}

/* a thin accent stripe across the top */
#project-info::before {
  content: "";
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 4px;
  background: #80D0FF;  /* your accent color */
}

/* Footer heading */
#project-info h4 {
  margin: 0 0 12px !important;
  font-size: 1.4rem !important;
  color: #fff !important;
  letter-spacing: 0.5px !important;
}

/* Footer paragraph */
#project-info p {
  margin: 0 auto !important;
  max-width: 700px !important;
  line-height: 1.6 !important;
  color: #ddd !important;
  font-size: 1rem !important;
}

/* optional: style any links you might add later */
#project-info a {
  color: #80D0FF !important;
  text-decoration: none !important;
}
#project-info a:hover {
  text-decoration: underline !important;
}

/* --------------------------------------------------
   7. App Header Styling
-------------------------------------------------- */
#app-header {
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #2a2c3d 0%, #1e1f2a 100%);
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
  padding: 16px 24px;
  margin-bottom: 32px;
}
#app-header span {
  font-size: 2.5rem;
  font-weight: 700;
  color: #80D0FF;
  letter-spacing: 1px;
}

"""
) as dashboard:
#    gr.Markdown(
#        """<span style="color:#80D0FF; font-size:36px; font-weight:600;">📖 NextRead</span>"""
#    )
    # wrap the header in its own div so we can style it
    gr.Markdown(
       """
       <div id="app-header">
         <span>📖 NextRead</span>
       </div>
       """,
       elem_id="app-header"
       )
    # Subheading 
    gr.Markdown(
        """<span style="font-size:18px; font-style:italic; color:#CCCCCC;">
        Welcome to smart book recommendations, powered by <b>semantic discovery</b>.
        </span>"""
    )
    # Featured Books
    gr.Markdown("<h3 style='color:#FFD700;'>🌟 Featured Books 🌟</h3>")
    featured_gallery = gr.Gallery(
        value=get_featured_books,
        show_label=False,
        columns=6,
        rows=2,
        elem_id="featured-gallery"
    )
    # User Input Block
    with gr.Row():
        user_query = gr.Textbox(
            label="✍️ Please enter a description of a book",
            placeholder="e.g., A Story about forgiveness...",
            lines=2
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="🔖 Select a category:", value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="🎭 Select an emotional tone:", value="All"
        )
    # Submit Button
    with gr.Row():
        submit_button = gr.Button("🔍 Find recommendations")

    # Output Block
    gr.Markdown(
        """<h2 id='recommendations-header' style='text-align:center; color:#80D0FF;'>✨ Recommendations ✨</h2>"""
    )
    output = gr.Gallery(
        label="📚 Recommended Books for You",
        show_label=True,
        columns=5,
        object_fit="cover",
        elem_id="gallery-box"
    )

    # Footer
    gr.Markdown(
        """
        ---  
        <div id="project-info">
          <h4>ℹ️ About NextRead</h4>
          <p>
            NextRead is a hands-on demo of a semantic book recommender built with 
            <b>LangChain</b>, <b>HuggingFace embeddings</b>, and <b>Gradio</b>.  
            Enter what you feel like reading, pick an emotion or category,  
            and voilà! your next great read is just a click away!
          </p>
        </div>
        """,
        elem_id="project-info"
    )
    # wire up the submit button
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
