def build_shap_summary_prompt(ctx: dict) -> str:
    return f"""
You are a senior manufacturing ML engineer.

Interpret the SHAP SUMMARY PLOT for a predictive model.

Feature importance (mean |SHAP| values):
{ctx['feature_importance']}

Top important features:
{ctx['top_features']}

Provide:
1. Key drivers of the target
2. Operational meaning of these drivers
3. Whether influence is strong or weak
4. How stable/robust the model seems
5. Recommended next steps to validate findings

Return STRICT JSON:
{{
 "insight": "",
 "confidence": 0.0,
 "suggested_next_steps": []
}}
"""


def build_shap_dependence_prompt(ctx: dict) -> str:
    return f"""
You are a senior manufacturing ML engineer.

Interpret the SHAP DEPENDENCE PLOT.

Feature analyzed: {ctx['feature']}

Sample SHAP values (first 50): {ctx['shap_values'][:50]}
Sample feature values (first 50): {ctx['feature_values'][:50]}
Correlation(feature, shap): {ctx['correlation']}

Provide:
1. Shape of relationship (linear / non-linear / threshold / saturation)
2. Whether effect is positive or negative
3. Whether interaction effects are visible
4. Operational meaning for process/plant reliability
5. Recommended next analysis

Return STRICT JSON:
{{
 "insight": "",
 "confidence": 0.0,
 "suggested_next_steps": []
}}
"""
