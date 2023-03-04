"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

from musiccollaborativefiltering.data import load_user_artists, ArtistRetriever


class ImplicitRecommender:  #מחלקה שממחשבת המלצה עבור משתמש נתון באמצעות סינון שיתופי תוך שימוש בספריית הimplicit
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
            self,
            artist_retriever: ArtistRetriever,
            implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:  #פעולה הלוקחת את מטריצת הuser artists ומשתמשת בה כדי לאמן את הimplicit model
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
        self,
        user_id: int, #הid של המשתמש לו נרצה להמליץ על אומן
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10, #מספר האומנים אותם נמליץ למשתמש
    ) -> Tuple[List[str], List[float]]: #שתי רשימות כאשר הראשונה היא רשימה של האמנים והשנייה היא רשימה של הציון של כל האומנים מהרשימה של האומנים
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(   #כאן אנו נעביר לפעולה את הuser id ואת המטריצה user artists, בנוסף למספר n שהוא האינדקס של אורך הרשימות שאותן נקבל בסוף
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id) #כאן נהפוך את הרשימה של הartist ids לרשימה של artist names באמצעות הפעולה artist retriever
            for artist_id in artist_ids
        ]
        return artists, scores #מחזיר את שתי הרשימות


if __name__ == "__main__":

    # כאן נרצה להעלות את הuser artist matrix
    user_artists = load_user_artists(Path("../lastfmdata/user_artists.dat")) #נשתמש כאן בפעולת הload user artist

    # כאן ניצור אובייקט שהוא הartist retriever  ונעלה את הartists
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../lastfmdata/artists.dat"))

    # ניצור אובייקט als באמצעות implicit
    implict_model = implicit.als.AlternatingLeastSquares(  #הפעולה לוקחת מספר של ארגומנטים, במקרה הזה הם מספר המימדים אליהם נרצה להפריד את מטריצת הuser artist, מספר האיטרציות, ומספר ההסדרות
        factors=50, iterations=5, regularization=0.01
    )

    # ניצור אובייקט של המלצה, התאמה וממליץ
    recommender = ImplicitRecommender(artist_retriever, implict_model)
    recommender.fit(user_artists) #כאן נאמן את הממליץ באמצעות העברת מטריצת הuser artists
    artists, scores = recommender.recommend(2, user_artists, n=10) #כעת ניתן להמליץ באמצעות שימוש במתודת הrecommend , באמצעות העברת הid של המשתמש, קובץ הuser artists, ומספר האומנים אותם נרצה שיומלצו

    # print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}" )    #כאן תודפס רשימה של האומנים והציון שהמערכת נתנה להם

