from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    @staticmethod
    def create():
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
