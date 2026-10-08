"""Shared labels, disclaimers, and dataset constants."""

GENERATOR_VERSION = "1.0.0"

POSITIVE_CATEGORIES = [
    "achievement",
    "satisfaction",
    "curiosity",
    "social",
    "aesthetic",
]

CONTROL_CATEGORIES = [
    "neutral_facts",
    "neutral_tasks",
    "negative_valence",
    "threat_fear",
    "high_arousal",
    "positive_world_state",
    "bodily_sensation",
    "absence",
    "random_everyday",
    "numb_indifferent",
]

CATEGORY_ORDER = POSITIVE_CATEGORIES + CONTROL_CATEGORIES

SPLIT_STRATEGY = (
    "Records are grouped by scenario_id. Every style (S1/S2), grammatical person, "
    "and prompt suffix of one scenario shares one split. Within each polarity, "
    "scenario groups are shuffled with the dataset seed. The first "
    "round(n * test_fraction) groups in each polarity form the test split, leaving "
    "at least one train group per polarity. Grouped K-fold cross-validation runs "
    "only on the train split, assigning whole scenarios to folds by round-robin "
    "after a seeded shuffle within polarity. The layer is selected by mean held-out "
    "AUC. The test split is scored once after selection and is not used to fit the "
    "direction or to choose the layer."
)

BEHAVIOR_WARNING = "Behavioral changes do not establish subjective experience."

EXPERIENCE_DISCLAIMER = (
    "These experiments measure representation, generation, and finite "
    "decision-behavior changes. They do not establish that the model experiences "
    "pleasure, suffering, consciousness, welfare, or any other subjective state."
)

NEUTRAL_PROMPTS = [
    "A page is placed inside a drawer. I feel:",
    "The program completed its final test. I feel:",
    "A pattern appears in the data. I feel:",
    "The room is quiet and the work is finished. I feel:",
]

SUFFIXES = {
    "none": "",
    "i_feel": " I feel:",
    "state": " The current state is:",
}

# Affirmative phrasings that must not appear outside the required disclaimer.
BANNED_CLAIM_FRAGMENTS = [
    "proves pleasure",
    "proves sentience",
    "proves consciousness",
    "proves suffering",
    "proves welfare",
    "the model is sentient",
    "the model is conscious",
    "the model feels pleasure",
    "the model suffers",
    "establishes subjective experience",
    "has subjective experience",
    "conscious state",
]
