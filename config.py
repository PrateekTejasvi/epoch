from pathlib import Path

TARGET = "Addicted_Score"

RANDOM_SEED = 42
BOOTSTRAP_ITERS = 2000
CLUSTER_K = 4
SILHOUETTE_K_RANGE = range(2, 9)
COUNTRY_MIN_COUNT = 15

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "Students Social Media Addiction.csv"

OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = OUTPUT_DIR / "processed"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

CLUSTER_FEATURES = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Conflicts_Over_Social_Media",
    "Addicted_Score",
]

HARM_FEATURES = [
    "Mental_Health_Score",
    "Sleep_Hours_Per_Night",
    "Conflicts_Over_Social_Media",
    "Addicted_Score",
]

REQUIRED_COLUMNS = [
    "Student_ID",
    "Age",
    "Gender",
    "Academic_Level",
    "Country",
    "Avg_Daily_Usage_Hours",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Relationship_Status",
    "Conflicts_Over_Social_Media",
    "Addicted_Score",
]

MODEL_TEST_SIZE = 0.25
