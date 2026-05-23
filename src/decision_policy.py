def apply_business_policy(probability_of_default: float) -> str:
    if probability_of_default < 0.10:
        return "Approve"
    elif probability_of_default <= 0.30:
        return "Review"
    return "Reject"
