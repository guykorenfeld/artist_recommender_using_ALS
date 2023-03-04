"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""

from pathlib import Path
import scipy
import pandas as pd


def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user artists file and return a user-artists matrix in csr
    fromat """     #הפעולה מקבלת קובץ מסוג user artist, שזה בעצם נתיב לקובץ באותו השם, ומחזירה מטריצה sparse מסוג csr

    user_artists = pd.read_csv(user_artists_file, sep="\t")  # מעלים את הuser artist באמצעות הספריה pandas (משתמשים בפונקצית read csv) ומשתמשים בt כמחיצה
    user_artists.set_index(["userID", "artistID"], inplace=True)  #   מגדירים את האינדקסים לuser id ול artist id (העמודות מתוך הקובץ user artist).
    coo = scipy.sparse.coo_matrix(      # יוצרים מטריצת coo, שהיא פורמט נוסף של מטריצת sparse.
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    # מחזירים את המטריצה לאחר שהופכים אותה למטריצת csr (זאת מכיוון שאנו נשתמש בספרייה שבה יש als implemantation שעובדת עם מטריצת csr כקלט
    return coo.tocsr()


class ArtistRetriever:
    """The ArtistRetriever class gets the artist name from the artist ID-."""
    """מקבלת את שם האמן מהאיי די שלו"""
    def __init__(self):
        self._artists_df = None # פעולה בונה היוצרת אובייקט מתוך הartist dataframe אותו קובעים כnone

    def get_artist_name_from_id(self, artist_id: int) -> str: #כאן מתקבל המספר של אומן ומחוזר string של שם האומן
        """Return the artist name from the artist ID."""
        return self._artists_df.loc[artist_id, "name"] #על מנת להחזיר את שם האומן אנו משתמשים בdataframe ובפעולת הloc בה מועבר הartist id

    def load_artists(self, artists_file: Path) -> None: #מתודה בה נעביר את הartists file
        """Load the artists file and stores it as a Pandas dataframe in a
        private attribute.
        """
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df #כאן שומרים את הartists file כתכונה (attribute) פרטית של הdataframe


if __name__ == "__main__":

     user_artists_matrix = load_user_artists( #מעלים את המטריצה של user artist
         Path("../lastfmdata/user_artists.dat")
         )

     print(user_artists_matrix) #מדפיסים את אותה המטריצה

     artist_retriever = ArtistRetriever() # artist retriever נוצר אובייקט מסוג
     artist_retriever.load_artists(Path("../lastfmdata/artists.dat")) #נעלה את הartist (זה נתיב לקובץ artist שנמצא בlastfmdata)
     artist = artist_retriever.get_artist_name_from_id(760) # הגדרת המשתנה artist באמצעות קריאה לפונקציה get artist name from id (מציאת האומן לפי הid שלו)
     print(artist) #מדפיס את שם האומן
