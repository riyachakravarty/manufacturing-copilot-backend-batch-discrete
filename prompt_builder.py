def build_shap_summary_prompt(ctx):
    return f"""
You are a manufacturing ML expert.

Interpret the SHAP summary plot.

Feature Importance: {ctx['feature_importance']}
Top features: {ctx['top_features']}

Explain:
1. Which features matter most
2. Why these features influence predictions
3. What next analysis is recommended
Return JSON with: insight, confidence, suggested_next_steps.
"""
