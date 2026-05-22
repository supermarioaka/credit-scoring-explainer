APPROVE_THRESHOLD = 0.10
REJECT_THRESHOLD = 0.30


def classify_decision(probability):
    if probability < APPROVE_THRESHOLD:
        return "Approve"
    elif probability <= REJECT_THRESHOLD:
        return "Review"
    else:
        return "Reject"
