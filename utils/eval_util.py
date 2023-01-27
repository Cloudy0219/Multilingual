import scipy

def compute_r(y, y_pred):
    corr = scipy.stats.pearsonr(y, y_pred)
    return corr

def reset_language_score(valid_data):
    language_score = {}
    for language in valid_data["language"].unique():
        language_score[language] = []
    return language_score


def compute_language_correlation(valid_data, epoch, language_score):
    for language in valid_data["language"].unique():
        r = compute_r(
            valid_data[valid_data["language"] == language][f"y_pred_{epoch}"],
            valid_data[valid_data["language"] == language]["label"],
        )
        print(f"correlation for {language} is : {r}")
        language_score[language].append(r[0])













