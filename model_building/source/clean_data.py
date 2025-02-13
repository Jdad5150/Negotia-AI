from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import us
import acora


def download_dataset():
    """
    This function downloads the LinkedIn Job Postings dataset from Kaggle and saves it to the data directory.
    """
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")

    api = KaggleApi()
    api.authenticate()

    dataset = "arshkon/linkedin-job-postings"
    save_path = os.path.join(os.getcwd(), "model_building", "data")
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading dataset: {dataset} to {save_path}")
    api.dataset_download_files(dataset, path=save_path, unzip=True)
    print("Download complete.")


def group_city_to_state(city):
    city = city.lower()
    if any(keyword.lower() in city for keyword in alabama_cities):  #
        return "Alabama"
    elif any(keyword.lower() in city for keyword in arkansas_cities):  #
        return "Arkansas"
    elif any(keyword.lower() in city for keyword in arizona_cities):  #
        return "Arizona"
    elif any(keyword.lower() in city for keyword in california_cities):  #
        return "California"
    elif any(keyword.lower() in city for keyword in colorado_cities):  #
        return "Colorado"
    elif any(keyword.lower() in city for keyword in conneticut_cities):  #
        return "Connecticut"
    elif any(keyword.lower() in city for keyword in virginia_cities):  #
        return "Virginia"
    elif any(keyword.lower() in city for keyword in florida_cities):  #
        return "Florida"
    elif any(keyword.lower() in city for keyword in georgia_cities):  #
        return "Georgia"
    elif any(keyword.lower() in city for keyword in hawaii_cities):  #
        return "Hawaii"
    elif any(keyword.lower() in city for keyword in iowa_cities):  #
        return "Iowa"
    elif any(keyword.lower() in city for keyword in idaho_cities):  #
        return "Idaho"
    elif any(keyword.lower() in city for keyword in illinois_cities):  #
        return "Illinois"
    elif any(keyword.lower() in city for keyword in indiana_cities):  #
        return "Indiana"
    elif any(keyword.lower() in city for keyword in kansas_cities):  #
        return "Kansas"
    elif any(keyword.lower() in city for keyword in kentucky_cities):  #
        return "Kentucky"
    elif any(keyword.lower() in city for keyword in louisiana_cities):  #
        return "Louisiana"
    elif any(keyword.lower() in city for keyword in massachusetts_cities):  #
        return "Massachusetts"
    elif any(keyword.lower() in city for keyword in michigan_cities):  #
        return "Michigan"
    elif any(keyword.lower() in city for keyword in minnesota_cities):  #
        return "Minnesota"
    elif any(keyword.lower() in city for keyword in missouri_cities):  #
        return "Missouri"
    elif any(keyword.lower() in city for keyword in north_carolina_cities):  #
        return "North Carolina"
    elif any(keyword.lower() in city for keyword in north_dakota_cities):  #
        return "North Dakota"
    elif any(keyword.lower() in city for keyword in nebraska_cities):  #
        return "Nebraska"
    elif any(keyword.lower() in city for keyword in new_mexico_cities):  #
        return "New Mexico"
    elif any(keyword.lower() in city for keyword in nevada_cities):  #
        return "Nevada"
    elif any(keyword.lower() in city for keyword in new_york_cities):  #
        return "New York"
    elif any(keyword.lower() in city for keyword in ohio_cities):  #
        return "Ohio"
    elif any(keyword.lower() in city for keyword in oklahoma_cities):  #
        return "Oklahoma"
    elif any(keyword.lower() in city for keyword in oregon_cities):  #
        return "Oregon"
    elif any(keyword.lower() in city for keyword in pennsylvania_cities):  #
        return "Pennsylvania"
    elif any(keyword.lower() in city for keyword in south_dakota_cities):  #
        return "South Dakota"
    elif any(keyword.lower() in city for keyword in tennessee_cities):  #
        return "Tennessee"
    elif any(keyword.lower() in city for keyword in texas_cities):  #
        return "Texas"
    elif any(keyword.lower() in city for keyword in utah_cities):  #
        return "Utah"
    elif any(keyword.lower() in city for keyword in vermont_cities):  #
        return "Vermont"
    elif any(keyword.lower() in city for keyword in washington_dc):  #
        return "District of Columbia"
    elif any(keyword.lower() in city for keyword in washington_cities):  #
        return "Washington"
    elif any(keyword.lower() in city for keyword in wisconsin_cities):  #
        return "Wisconsin"
    elif any(keyword.lower() in city for keyword in west_virginia_cities):  #
        return "West Virginia"
    else:
        return "Unknown"  # For cities that don't fit any predefined state


def convert_state_abbreviation(state_abbr):
    """
    Converts state abbreviations to full names. If already a full name, it remains unchanged.
    """
    # Manual dictionary for states, including cases where multiple states are listed
    manual_states = {
        "DC/MD/VA": "District of Columbia, Maryland, Virginia",
        "MO/IL": "Missouri, Illinois",
        "NC/SC": "North Carolina, South Carolina",
        "TN/GA": "Tennessee, Georgia",
        "TN/VA": "Tennessee, Virginia",
        "PA/NJ/DE": "Pennsylvania, New Jersey, Delaware",
        "MO/KS": "Missouri, Kansas",
        "DC": "District of Columbia",
    }

    # Check manual mappings first
    if state_abbr in manual_states:
        return manual_states[state_abbr]

    # Use the 'us' library for standard conversions
    state_obj = us.states.lookup(state_abbr)
    return state_obj.name if state_obj else state_abbr  # Keep unchanged if not found


def group_job_titles(title):
    title = title.lower()

    # Group by job categories
    if "engineer" in title or "software" in title or "developer" in title:
        if any(
            keyword in title
            for keyword in [
                "software",
                "frontend",
                "front end",
                "back end",
                "backend",
                "fullstack",
                "ui",
                "ux",
            ]
        ):  #
            return "Software Engineering"
        elif "data" in title:
            return "Data Engineer"
        elif "mechanical" in title:
            return "Mechanical Engineering"
        elif "quality" in title:  #
            return "Quality Engineering"
        elif "aerospace" in title:  #
            return "Aerospace Engineering"
        elif "web" in title or ".net" in title:  #
            return "Web Developer"
        else:
            return "General Engineering"
    elif (
        "programmer" in title
        or "front end" in title
        or "graphic" in title
        or "graphics" in title
        or "it" in title
        or "developer" in title
    ):
        return "Software Engineering"
    elif (
        "quality assurance" in title
        or "auditor" in title
        or "claims" in title
        or "inspector" in title
        or "inspectors" in title
        or "quality" in title
        or "qc" in title
        or "qa" in title
    ):
        return "Quality Assurance"
    elif "data" in title:
        return "Data Science"
    elif "marketing" in title or "seo" in title or "social media" in title:
        return "Marketing"
    elif "sales" in title or "business development" in title:
        return "Sales"
    elif "product" in title:
        return "Product Management"
    elif (
        "hr" in title
        or "recruiter" in title
        or "admissions" in title
        or "admitting" in title
        or "appointment" in title
        or "human resource" in title
        or "human resources" in title
    ):
        return "Human Resources"
    elif (
        "nurse" in title
        or "nursing" in title
        or "clinic" in title
        or "clinical" in title
        or "health" in title
        or "rn" in title
        or "cna" in title
        or "care" in title
        or "lpn" in title
        or "med" in title
        or "aide" in title
    ):
        return "Nurse"
    elif (
        "doctor" in title
        or "medical" in title
        or "physician" in title
        or "medicine" in title
        or "phlebotomist" in title
        or "surgical" in title
        or "Chiroprator" in title
        or "surgeon" in title
        or "psychiatrist" in title
        or "psychologist" in title
        or "neurologist" in title
    ):
        return "Doctor"
    elif "pharmacist" in title:
        return "Pharmacist"
    elif "dental" in title or "dentist" in title:
        return "Dentist"
    elif (
        "teacher" in title
        or "training" in title
        or "instructor" in title
        or "tutor" in title
        or "trainer" in title
    ):
        return "Teacher"
    elif "counselor" in title or "counsel" in title:
        return "Counselor"
    elif "professor" in title:
        return "Professor"
    elif "academic" in title or "student" in title:
        return "Education"
    elif "therapist" in title or "behavioral" in title or "behavior" in title:
        return "Therapist"
    elif "law" in title or "legal" in title or "attorney" in title:
        return "Legal"
    elif (
        "technician" in title
        or "operator" in title
        or "journeyman" in title
        or "techncian" in title
        or "machinist" in title
        or "manufacturing" in title
        or "fabricator" in title
        or "maintenance" in title
    ):
        return "Technician"
    elif any(
        keyword in title
        for keyword in [
            "accountant",
            "analyst",
            "accounting",
            "financial",
            "accounts",
            "account",
            "wealth",
            "cfo",
            "finance",
            "collection",
            "collections",
            "controller",
            "underwriter",
            "estimator",
            "insurance",
        ]
    ):
        return "Finance"
    elif (
        "hospitality" in title
        or "hotel" in title
        or "chef" in title
        or "waiter" in title
        or "restaurant" in title
        or "baker" in title
        or "bakery" in title
        or "barista" in title
        or "bar" in title
        or "cook" in title
        or "dishawasher" in title
    ):
        return "Hospitality"
    elif (
        "customer" in title
        or "service" in title
        or "support" in title
        or "cashier" in title
    ):
        return "Customer Service"
    elif (
        "administrative" in title
        or "admin" in title
        or "clerk" in title
        or "planner" in title
    ):
        return "Administration"
    elif (
        "laborer" in title
        or "utility" in title
        or "assembly" in title
        or "assembler" in title
        or "labor" in title
        or "handler" in title
        or "warehouse" in title
        or "picker" in title
        or "selector" in title
    ):
        return "Laborer"
    elif "worship" in title:
        return "Religous"
    elif "coordinator" in title:
        return "Coordinator"
    elif "electrician" in title or "electrical" in title:
        return "Electrician"
    elif "art" in title:
        return "Artist"
    elif "security" in title:
        return "Security"
    elif "lead" in title or "leader" in title or "foremane" in title:
        return "Team Lead"
    elif "architect" in title or "architectural" in title:
        return "Architect"
    elif "banker" in title or "bank" in title or "banking" in title:
        return "Banker"
    elif "driver" in title:
        return "Driver"
    elif (
        "manager" in title
        or "management" in title
        or "director" in title
        or "president" in title
        or "vice-president" in title
        or "vice president" in title
        or "ambassador" in title
        or "executive" in title
        or "chief" in title
        or "superintendent" in title
        or "supervisor" in title
        or "advisor" in title
    ):
        return "Manager"
    elif "coach" in title or "coaching" in title:
        return "Coach"
    elif "assistant" in title or "receptionist" in title or "attendant" in title:
        return "Assistant"
    elif "veterinarian" in title:
        return "Veterinarian"
    elif "associate" in title:
        return "Associate"
    elif (
        "auto" in title
        or "automotive" in title
        or "mechanic" in title
        or "collision" in title
    ):
        return "Mechanic"
    elif "firefighter" in title or "fire" in title:
        return "Firefighter"
    elif (
        "police" in title
        or "detective" in title
        or "investigator" in title
        or "officer" in title
        or "security" in title
        or "securities" in title
    ):
        return "Police"
    elif (
        "scientist" in title
        or "chemist" in title
        or "microbiologist" in title
        or "biologist" in title
    ):
        return "Scientist"
    elif "bookkeeper" in title or "library" in title or "librarian" in title:
        return "Librarian"
    elif "specialist" in title:
        return "Specialist"
    elif (
        "business" in title
        or "buyer" in title
        or "merchant" in title
        or "merchandiser" in title
    ):
        return "Business"
    elif "designer" in title or "design" in title:
        return "Designer"
    elif "consultant" in title:
        return "Consultant"
    elif "creator" in title:
        return "Content Creator"
    elif "pilot" in title or "helicopter" in title:
        return "Pilot"
    elif "technologist" in title:
        return "Technologist"
    elif (
        "cleaner" in title
        or "janitor" in title
        or "custodian" in title
        or "housekeeper" in title
        or "housekeeping" in title
    ):
        return "Custodian"
    elif "agent" in title:
        return "Agent"
    elif "environmental" in title or "environment" in title:
        return "Environmental"
    elif "intern" in title:
        return "Intern"
    elif "welder" in title or "weld" in title:
        return "Welder"
    elif "representative" in title:
        return "Representative"

    else:
        return "Other"



if __name__ == "__main__":

    # Make sure you are in the correct directory
    # you should be in 'model_building'
    print(os.getcwd())

    needs_data = input("Do you need to download the data? Press 1 for yes, 0 for no: ")

    if needs_data == "1":
        download_dataset()
    else:
        print("Data download skipped.")

    file_name = "postings.csv"

    df = pd.read_csv("data/" + file_name)
    print(df.info())

    in_features = ["title", "location", "work_type", "formatted_experience_level"]
    out_feature = "normalized_salary"

    df = df[in_features + [out_feature]]

    # Clean Experience Level
    df["formatted_experience_level"] = df["formatted_experience_level"].fillna(
        "Not Specified"
    )

    # Make the strings look nice
    df['work_type'] = df['work_type'].str.replace('_', ' ').str.title()

    # Remove rows with missing output
    df = df.dropna(subset=["normalized_salary"])
    df = df[df["location"] != "United States"]

    # Clean location
    alabama_cities = ["Mobile", "Dothan"]
    arkansas_cities = ["Little Rock"]
    arizona_cities = ["Phoenix", "Tucson", "Flagstaff"]
    california_cities = [
        "Los Angeles",
        "San Francisco Bay",
        "San Diego",
        "Sacramento",
        "Fresno",
        "Modesto-Merced",
        "Greater Chico",
        "San Luis Obispo",
    ]
    colorado_cities = [
        "Denver",
        "Colorado Springs",
        "Fort Collins",
        "Pueblo",
        "Grand Junction",
    ]
    conneticut_cities = ["Hartford"]
    virginia_cities = ["Blacksburg-Christiansburg-Radford", "Greater Richmond Region"]
    florida_cities = [
        "Miami",
        "Orlando",
        "Tampa",
        "Jacksonville",
        "Fort Lauderdale",
        "Tallahassee",
        "Pensacola",
        "Cape Coral",
        "Fort Walton",
    ]
    georgia_cities = ["Atlanta", "Augusta", "Savannah", "Macon", "Roswell"]
    hawaii_cities = ["Honolulu", "Maui"]
    iowa_cities = ["Des Moines", "Waterloo"]
    idaho_cities = ["Boise"]
    illinois_cities = ["Chicago", "Peoria"]
    indiana_cities = ["Indianapolis", "Mishawaka", "Fort Wayne", "Bloomington"]
    kansas_cities = ["Topeka"]
    kentucky_cities = ["Louisville", "Lexington"]
    louisiana_cities = ["New Orleans", "Baton Rouge"]
    massachusetts_cities = ["Boston"]
    michigan_cities = ["Detroit", "Grand Rapids", "Lansing"]
    minnesota_cities = ["Minneapolis"]
    missouri_cities = ["St. Louis", "Kansas City"]
    north_carolina_cities = [
        "Charlotte",
        "Raleigh",
        "Greensboro",
        "Morehead",
        "Greenville",
        "Mount-Wilson",
        "Asheville",
        "Wilmington",
        "Goldsboro",
    ]
    north_dakota_cities = ["Bismarck"]
    nebraska_cities = ["Omaha"]
    new_mexico_cities = ["Albuquerque"]
    nevada_cities = ["Las Vegas", "Reno"]
    new_york_cities = ["New York City", "Buffalo-Niagara Falls", "Syracuse", "Utica"]
    ohio_cities = ["Cleveland", "Cincinnati", "Youngstown"]
    oklahoma_cities = ["Oklahoma", "Tulsa", "Lawton", "Greater Enid"]
    oregon_cities = ["Greater Bend", "Greater Eugene-Springfield"]
    pennsylvania_cities = ["Pittsburgh", "Scranton", "Erie-Meadville", "Philadelphia"]
    south_dakota_cities = ["Sioux Falls"]
    tennessee_cities = ["Knoxville", "Nashville", "Chattanooga", "Memphis", "Bristol"]
    texas_cities = [
        "Dallas-Fort Worth Metroplex",
        "Dallas",
        "Fort Worth",
        "Houston",
        "Lubbock",
        "Beaumont-Port Arthur",
        "Corpus Christi",
        "McAllen",
        "College Station-Bryan",
    ]
    utah_cities = ["Salt Lake City"]
    vermont_cities = ["Burlington"]
    washington_dc = ["Washington DC"]
    washington_cities = ["Seattle", "Bellingham", "Walla"]
    wisconsin_cities = [
        "La Crosse-Onalaska",
        "Eau Claire-Menomonie",
        "Greater Madison",
        "Greater Milwaukee",
        "Appleton-Oshkosh-Neenah",
    ]
    west_virginia_cities = ["Morgantown"]



    # Create a copy of the location feature, for engineer and split into city, state, country
    location = df["location"].str.split(",", expand=True)
    location.columns = ["City", "State", "Country"]

    # Drop country and remove extra words and spaces from the state column
    location = location.drop("Country", axis=1)
    location["State"] = location["State"].str.replace(
        r"(Metropolitan Area|Area)\s*$", "", regex=True
    )
    location["State"] = location["State"].str.strip()

    # If the 'State' is empty, move the 'City' to the 'State'
    location["State"] = location.apply(
        lambda row: row["City"] if row["State"] == "United States" else row["State"],
        axis=1,
    )

    # Remove extra words again
    location["City"] = location["City"].str.replace(
        r"(Metropolitan Area|Area)\s*$", "", regex=True
    )


    for index, row in location.iterrows():
        if pd.isna(row["State"]) or row["State"] == "Unknown":
            location.at[index, "State"] = group_city_to_state(row["City"])

    # Convert Abbreviations to Full String Name
    location["State"] = location["State"].apply(convert_state_abbreviation)

    # Move clean location to the main DataFrame
    df["location"] = location["State"]

    cleaning_df = pd.DataFrame()
    cleaning_df["title_copy"] = df.title
    cleaning_df["category"] = cleaning_df["title_copy"].apply(group_job_titles)
    df["title"] = cleaning_df["category"]
    df = df[df["title"] != "Other"]

    df = df.rename(columns={'title' :'job_title', 'location' :'state','work_type':'work_type','normalized_salary':'salary','formatted_experience_level':'experience_level'})
    
    df.to_parquet("data/cleaned_data.parquet", index=False)
