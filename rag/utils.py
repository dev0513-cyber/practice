def format_prediction_summary(pred):
    lines = [
        "--- ML Model Prediction ---",
        f"Current Price : ${pred['current_price']:,.2f}",
        f"Predicted Price: ${pred['predicted_price']:,.2f}",
        f"Expected Change: {pred['price_change_pct']:+.2f}%",
        f"Prob of Up Move: {pred['direction_prob_up']:.1f}%",
        f"Signal         : {pred['signal']}",
        "---------------------------",
    ]
    return "\n".join(lines)


def build_llm_prompt(user_question, ml_summary, rag_context):
    prompt = f"""You are a crypto analysis assistant. Answer the user's question using the ML prediction and context provided.

User Question: {user_question}

ML Model Output:
{ml_summary}

Relevant Knowledge:
{rag_context}

Provide a concise, informative answer. Do not make strong financial guarantees. Be balanced and factual.
"""
    return prompt
