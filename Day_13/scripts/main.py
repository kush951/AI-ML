"""
main.py
=======
Master entry point — runs the complete MeetMux ML pipeline:

  Step 1 : Generate synthetic attendee dataset
  Step 2 : Preprocess + Polynomial Features + split
  Step 3 : Train Random Forest classifier
  Step 4 : Evaluate — Confusion Matrix + Feature Importance
  Step 5 : Live prediction demo

Run:
    python main.py
"""

import time

from data_generator import generate_users, generate_pairs, save_datasets
from preprocessor   import Preprocessor
from model          import MatcherModel
from evaluator      import Evaluator
from predictor      import Predictor


BANNER = """
╔══════════════════════════════════════════════════════╗
║         MeetMux — Event Networking Matcher           ║
║         AI/ML Capstone  |  Mux Intelligence          ║
╚══════════════════════════════════════════════════════╝
"""


def run_pipeline():
    print(BANNER)
    t0 = time.time()

    # Step 1: Data generation
    print("------------Step 1 / 5 : Generating dataset---------")
    users = generate_users()
    pairs = generate_pairs(users)
    save_datasets(users, pairs)
    print(f"   Users: {len(users)}  |  Pairs: {len(pairs)}  "
          f"|  Match rate: {pairs['label'].mean()*100:.1f}%")

    # Step 2: Preprocessing
    print("\n-------Step 2 / 5 : Preprocessing-----------")
    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(pairs)
    prep.save()

    # Step 3: Training
    print("\n-----------Step 3 / 5 : Training Random Forest---------")
    model = MatcherModel()
    model.train(X_train, y_train)
    model.save()

    #Step 4:Evaluation
    print("\n--- Step 4 / 5 : Evaluation--- ")
    ev = Evaluator(model, prep)
    metrics = ev.evaluate(X_test, y_test)
    ev.plot_all(save=True)

    #Step 5:Live prediction demo
    print("\n-------Step 5 / 5 : Live Prediction Demo --------")
    predictor = Predictor(model, prep)

    demo_pairs = [
        (
            {"domain": "AI/ML",       "goal": "Find co-founder",
             "experience": 4,          "comm_style": "Direct",
             "location": "Bangalore"},
            {"domain": "Data Science", "goal": "Learn & grow",
             "experience": 3,           "comm_style": "Collaborative",
             "location": "Bangalore"},
        ),
        (
            {"domain": "Frontend",    "goal": "Get hired",
             "experience": 1,          "comm_style": "Storyteller",
             "location": "Mumbai"},
            {"domain": "Backend",      "goal": "Hire talent",
             "experience": 8,           "comm_style": "Analytical",
             "location": "Remote"},
        ),
        (
            {"domain": "Product",     "goal": "Build network",
             "experience": 6,          "comm_style": "Collaborative",
             "location": "Delhi"},
            {"domain": "Design",       "goal": "Build network",
             "experience": 5,           "comm_style": "Storyteller",
             "location": "Delhi"},
        ),
    ]

    for user_a, user_b in demo_pairs:
        result = predictor.predict(user_a, user_b)
        Predictor.print_result(user_a, user_b, result)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    print(f"\n{'═'*54}")
    print(f"  Pipeline complete in {elapsed}s")
    print(f"  Accuracy : {metrics['accuracy']}%  |  "
          f"F1 : {metrics['f1']}%")
    print(f"  Plot saved → outputs/evaluation.png")
    print(f"  Model saved → models/random_forest.joblib")
    print(f"{'═'*54}\n")


if __name__ == "__main__":
    run_pipeline()
