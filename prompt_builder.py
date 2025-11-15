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
def build_shap_dependence_prompt(ctx: dict) -> str:
    return f"""
You are a manufacturing ML expert.

Interpret the SHAP dependence plot.

Feature: {ctx['feature']}
Sample SHAP values: {ctx['shap_values'][:50]}
Sample feature values: {ctx['feature_values'][:50]}
Correlation with model output: {ctx['correlation']}

Explain clearly:
1. What is the shape of the relationship? (linear / non-linear / threshold / saturation)
2. Is the feature positively or negatively influencing the prediction?
3. Any visible interactions in the scatter?
4. What does this mean operationally for the process?
5. What next step should the engineer take?

Return STRICT JSON:
{{
 "insight": "",
 "confidence": 0.0,
 "suggested_next_steps": []
}}
"""
