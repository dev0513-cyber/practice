from data_loader import fetch_bitcoin_prices
from features import add_features
from model import train_models, load_models, models_exist, predict
from rag import build_vector_store, retrieve_context
from utils import format_prediction_summary, build_llm_prompt
from ollama_client import generate_ollama_response


def main():
    print("Fetching Bitcoin price data...")
    df_raw = fetch_bitcoin_prices(days=365)
    print(f"Loaded {len(df_raw)} days of data.")

    print("Computing features...")
    df = add_features(df_raw)

    if models_exist():
        print("Loading existing models...")
        price_model, dir_model, scaler = load_models()
    else:
        print("Training models...")
        price_model, dir_model, scaler = train_models(df)

    pred = predict(df, price_model, dir_model, scaler)
    ml_summary = format_prediction_summary(pred)
    print("\n" + ml_summary)

    print("Building vector store...")
    vector_store = build_vector_store()
    print("Vector store ready.\n")

    print("Crypto AI Assistant ready. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        context = retrieve_context(vector_store, question)
        prompt = build_llm_prompt(question, ml_summary, context)

        print("\nGenerating answer...\n")
        answer = generate_ollama_response(prompt)
        print("Answer:", answer)
        print()


if __name__ == "__main__":
    main()
