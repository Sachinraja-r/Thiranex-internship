import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"D:\work\internship\1\NetFlix.csv"

df = pd.read_csv(file_path)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DATASET SHAPE =====")
print(df.shape)


print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

for col in ["director", "cast", "country"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

if "rating" in df.columns:
    df["rating"] = df["rating"].fillna(df["rating"].mode()[0])

print("\nDuplicates before removal:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates after removal:", df.duplicated().sum())

if "date_added" in df.columns:

    df["date_added"] = pd.to_datetime(
        df["date_added"],
        format="%d-%b-%y",
        errors="coerce"
    )

    print("\nMissing dates after conversion:")
    print(df["date_added"].isna().sum())

    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    print("\nYear Added Counts:")
    print(df["year_added"].value_counts().sort_index())


print("\n===== STATISTICS =====")
print(df.describe(include="all"))


if "type" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="type"
    )

    plt.title("Movies vs TV Shows")
    plt.xlabel("Type")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

if "country" in df.columns:

    top_countries = (
        df["country"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=top_countries.values,
        y=top_countries.index
    )

    plt.title("Top 10 Content Producing Countries")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")

    plt.tight_layout()
    plt.show()

if "year_added" in df.columns:

    year_data = (
        df["year_added"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        x=year_data.index,
        y=year_data.values,
        marker="o"
    )

    plt.title("Netflix Content Added Per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Titles")

    plt.grid(True)

    plt.tight_layout()
    plt.show()

if "rating" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.countplot(
        y="rating",
        data=df,
        order=df["rating"].value_counts().index
    )

    plt.title("Content Rating Distribution")

    plt.tight_layout()
    plt.show()

if "release_year" in df.columns:

    plt.figure(figsize=(12, 6))

    sns.histplot(
        df["release_year"],
        bins=30,
        kde=True
    )

    plt.title("Release Year Distribution")
    plt.xlabel("Release Year")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

genre_column = None

if "genres" in df.columns:
    genre_column = "genres"

elif "listed_in" in df.columns:
    genre_column = "listed_in"

if genre_column:

    genre_counts = (
        df[genre_column]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=genre_counts.values,
        y=genre_counts.index
    )

    plt.title("Top 10 Genres")
    plt.xlabel("Count")
    plt.ylabel("Genre")

    plt.tight_layout()
    plt.show()



print("\n===== PROJECT INSIGHTS =====")

if "type" in df.columns:
    print("\nMovies vs TV Shows:")
    print(df["type"].value_counts())

if "country" in df.columns:
    print("\nTop 5 Countries:")
    print(df["country"].value_counts().head())

if "rating" in df.columns:
    print("\nTop Ratings:")
    print(df["rating"].value_counts().head())

if "release_year" in df.columns:
    print("\nLatest Release Year:")
    print(df["release_year"].max())

    print("\nOldest Release Year:")
    print(df["release_year"].min())

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")