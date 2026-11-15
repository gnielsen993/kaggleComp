"""Stateless, row-level features shared by the Titanic experiments."""


def engineer_features(frame):
    """Return a copy with family size, alone status, and grouped name title."""
    result = frame.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    title = result["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    result["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Other")
    return result


def engineer_extended_features(frame):
    """Add label-free cabin information and a fare-per-family-member proxy."""
    result = engineer_features(frame)
    cabin = result["Cabin"].fillna("").str.strip()
    result["HasCabin"] = cabin.ne("").astype(int)
    result["Deck"] = cabin.str.extract(r"^([A-Za-z])", expand=False).str.upper().fillna("Unknown")
    result["FarePerPerson"] = result["Fare"] / result["FamilySize"].where(result["FamilySize"] > 0)
    return result
